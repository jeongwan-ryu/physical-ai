// XIAO ESP32S3 Sense: camera(RGB565 QVGA) -> HandDetect -> HandGestureRecognizer
// -> stdout(USB-Serial/JTAG)에 한 줄 JSON:  {"g":"like","s":0.97,"x":0.42,"seq":1}
// 로그(ESP_LOG)도 같은 포트로 나가므로 PC 쪽은 '{'로 시작하는 줄만 사용.
// 카메라 영상도 한 줄로:  IMG:<base64 JPEG>   (코어 1 sender 태스크가 비어 있을 때마다)
// 속도: 손 검출이 프레임당 ~125ms로 병목. 손을 추적 중이면 DET_EVERY 프레임마다만 검출하고
//       사이 프레임은 직전 박스로 제스처 분류만 한다. 손이 없을 땐 매 프레임 검출.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <list>

#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "img_converters.h"
#include "mbedtls/base64.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "hand_detect.hpp"
#include "hand_gesture_recognition.hpp"

static const char *TAG = "gesture_cam";
static const int DET_EVERY = 3;        // 손 추적 중 검출 주기(프레임). 1이면 매번 검출(느리지만 박스 정확)
static const int IMG_QUALITY = 40;     // JPEG 품질 0~100
static const size_t B64_CAP = 64 * 1024;

static QueueHandle_t img_q;                 // 추론 루프 -> sender 로 넘기는 fb (길이 1)
static std::atomic<bool> img_busy{false};  // sender가 fb를 들고 있는 동안 true

// 코어 1: fb를 JPEG -> base64로 바꿔 "IMG:...\n" 한 줄 출력 후 fb 반환. 실패하면 건너뜀(영상은 디버그용)
// 한 줄을 fwrite 한 번으로 써야 코어 0의 JSON printf가 줄 중간에 끼어들지 않는다
static void sender_task(void *)
{
    unsigned char *line = (unsigned char *)heap_caps_malloc(B64_CAP, MALLOC_CAP_SPIRAM);
    camera_fb_t *fb;
    while (true) {
        xQueueReceive(img_q, &fb, portMAX_DELAY);
        uint8_t *jpg = nullptr;
        size_t jpg_len = 0, b64_len = 0;
        if (line && frame2jpg(fb, IMG_QUALITY, &jpg, &jpg_len)) {
            memcpy(line, "IMG:", 4);
            if (mbedtls_base64_encode(line + 4, B64_CAP - 5, &b64_len, jpg, jpg_len) == 0) {
                line[4 + b64_len] = '\n';
                fwrite(line, 1, 5 + b64_len, stdout);
            }
            free(jpg);
        }
        esp_camera_fb_return(fb);
        img_busy = false;
    }
}

static esp_err_t camera_init()
{
    camera_config_t c = {};
    c.pin_pwdn = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;
    c.pin_xclk = XCLK_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_d7 = Y9_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;
    c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;
    c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;
    c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d0 = Y2_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href = HREF_GPIO_NUM;
    c.pin_pclk = PCLK_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_RGB565;
    c.frame_size = FRAMESIZE_QVGA;  // 320x240. 손 검출 입력(224x224)보다 살짝 큼
    c.fb_count = 3;  // 추론 1 + sender 1 + 드라이버 1
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;  // 추론이 느려도 항상 최신 프레임
    return esp_camera_init(&c);
}

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(camera_init());
    img_q = xQueueCreate(1, sizeof(camera_fb_t *));
    xTaskCreatePinnedToCore(sender_task, "img_sender", 4096, nullptr, 4, nullptr, 1);
    HandDetect detect;
    HandGestureRecognizer recognizer;
    ESP_LOGI(TAG, "ready");

    uint32_t seq = 0;
    int64_t t0 = esp_timer_get_time();
    int64_t us_cam = 0, us_det = 0, us_cls = 0;  // 30프레임 누적 단계별 시간
    int n_det = 0, n_img = 0;
    std::list<dl::detect::result_t> track;  // 추적 중인 손 박스 (비었으면 손 없음)
    int since_det = 0;
    while (true) {
        int64_t ta = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "frame grab failed");
            continue;
        }
        // esp32-camera의 RGB565는 big-endian. 색이 이상하면 RGB565LE로 바꿔볼 것
        dl::image::img_t img = {.data = fb->buf,
                                .width = (uint16_t)fb->width,
                                .height = (uint16_t)fb->height,
                                .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE};

        int64_t tb = esp_timer_get_time();
        if (track.empty() || ++since_det >= DET_EVERY) {
            std::list<dl::detect::result_t> &hands = detect.run(img);
            track.clear();
            if (!hands.empty()) track.push_back(hands.front());  // 점수 가장 높은 손 하나만
            since_det = 0;
            n_det++;
        }
        int64_t tc = esp_timer_get_time();
        if (track.empty()) {
            printf("{\"g\":\"no_hand\",\"s\":1.00,\"x\":-1,\"seq\":%lu}\n", (unsigned long)seq);
        } else {
            std::vector<dl::cls::result_t> res = recognizer.recognize(img, track);
            const std::vector<int> &b = track.front().box;
            float x = (b[0] + b[2]) / 2.0f / img.width;
            if (res.empty()) {
                printf("{\"g\":\"no_gesture\",\"s\":0.00,\"x\":%.2f,\"seq\":%lu}\n", x, (unsigned long)seq);
            } else {
                printf("{\"g\":\"%s\",\"s\":%.2f,\"x\":%.2f,\"seq\":%lu}\n", res[0].cat_name, res[0].score, x,
                       (unsigned long)seq);
            }
        }
        us_cam += tb - ta, us_det += tc - tb, us_cls += esp_timer_get_time() - tc;
        if (!img_busy) {  // sender가 놀고 있으면 이 fb를 넘기고, 반환은 sender가 한다
            img_busy = true;
            xQueueSend(img_q, &fb, 0);
            n_img++;
        } else {
            esp_camera_fb_return(fb);
        }

        if (++seq % 30 == 0) {
            int64_t t1 = esp_timer_get_time();
            ESP_LOGI(TAG, "%.1f fps | cam %lld det %lld cls %lld ms/frame | det %d/30 img %d/30", 30 * 1e6 / (t1 - t0),
                     us_cam / 30000, us_det / 30000, us_cls / 30000, n_det, n_img);
            us_cam = us_det = us_cls = 0;
            n_det = n_img = 0;
            t0 = t1;
        }
    }
}
