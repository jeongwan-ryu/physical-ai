# 분실물 검출 무인이동체 — 전처리 코드 (Raspberry Pi 5 버전)

`report/nooki_전처리_기법_및_순서_근거.md`(Raspberry Pi 5 버전)의 설계를 그대로 구현한 코드입니다.
학습 데이터 생성(PC)과 로봇 온보드 추론(Pi)이 **같은 전처리 코드**(`preprocess.py`)를 씁니다.

## 하드웨어·소프트웨어 가정

| 구분 | 내용 |
|---|---|
| 보드 | Raspberry Pi 5 (8GB), 액티브 쿨러 |
| AI 가속기 | Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L). 26 TOPS(Hailo-8)도 그대로 동작 |
| 카메라 | Camera Module 3 Wide (IMX708, 수평 102°, 롤링셔터, 오토포커스), 1280×720 출력 |
| LED | RP1 하드웨어 PWM (sysfs), 20kHz 이상 |
| OS·라이브러리 | Raspberry Pi OS Bookworm 64bit, Picamera2/libcamera, OpenCV(CPU), HailoRT(`hailo-all`) |

## 파일 구성 — 실행 환경 기준으로 4개

```
├── preprocess.py      PC · Pi 공용   ★ 핵심. 설정값 + ②~⑧ 전처리·후처리 + 파이프라인 조립
├── rpi_hardware.py    Pi 전용         ① 카메라(Picamera2) · LED PWM · Hailo 검출기
├── offline_tools.py   PC 전용         캘리브레이션 · 중복 제거 · 증강 · 학습 데이터셋 · Hailo 보정 세트
├── run_demo.py        실행 진입점     셀프테스트 · 녹화 영상 데모 · Pi 실시간 실행
├── train_hyp.yaml, data.yaml          YOLO 학습 설정
└── requirements.txt
```

**왜 이렇게 나눴나:** 파일을 나누는 기준을 "주제"가 아니라 **"어디서 실행되나"**로 잡았습니다.
- `preprocess.py`는 학습 데이터를 만들 때(PC)와 로봇이 추론할 때(Pi) **같은 코드를 써야** 합니다(학습-추론 불일치 방지). 그래서 공용 코드는 한 파일에만 둡니다.
- `rpi_hardware.py`는 picamera2·Hailo처럼 **Pi에만 설치되는 패키지**를 씁니다. 공용 파일과 섞으면 PC에서 import 오류가 나거나 "Pi인가?" 분기가 곳곳에 생깁니다.
- `offline_tools.py`는 **미리 PC에서 한 번씩** 하는 작업이라, 로봇 실시간 코드를 읽을 때 방해되지 않게 뺐습니다.

**공부 순서 추천:** `preprocess.py`를 위에서 아래로 읽으면 실제 데이터가 흐르는 순서(② LED → ③ 품질 → ④ 크롭 → ⑤ 플랫필드 → ⑥ CLAHE → ⑦ 타일링 → ⑧ 후처리 → 파이프라인 조립)와 같습니다. 그다음 `run_demo.py`의 `selftest()`를 실행하면서 각 단계가 어떻게 검사되는지 보고, `rpi_hardware.py` → `offline_tools.py` 순서로 읽으면 됩니다.

## Jetson 버전에서 바뀐 점

| 항목 | 이전 (Jetson) | 지금 (Raspberry Pi 5) |
|---|---|---|
| 노이즈 제거 | `tnr.py` (VPI TNR) | **삭제.** 카메라 ISP가 처리 (`RpiCameraConfig.noise_reduction`) |
| CLAHE | `cv2.cuda` (GPU) | CPU `cv2.createCLAHE`. 학습(PC)과 결과가 완전히 같아짐 |
| 플랫필드 | 선형 변환 LUT 두 번 + 곱셈 | 게인을 감마 영역으로 미리 변환해 `cv2.multiply` 한 번 (오차 ≤ 1) |
| 과노출 판정 | numpy `max(axis=2)` | OpenCV `split/max/compare/countNonZero` (PC 기준 11ms → 1ms) |
| 카메라 제어 | 없음 (드라이버 몫) | `rpi_hardware.py`의 `RpiCamera` 추가: Picamera2 컨트롤 + 메타데이터 → `FrameMeta` |
| LED | 없음 | `rpi_hardware.py`의 `LedPwm` 추가: RP1 하드웨어 PWM |
| 모델 실행 | TensorRT `.engine` | **Hailo `.hef`** (`rpi_hardware.HailoDetector`) |
| INT8 보정 | TensorRT 캘리브레이션 | `offline_tools.py calib-set` → Hailo DFC 컴파일 |
| 카메라 기하 | 수평 120° (fx≈370) | 수평 102° (fx≈518). 물체가 약 1.4배 크게 찍힘 |

## 사용 순서

### 1. 셀프테스트 (PC 또는 Pi, 카메라·모델 없이)

```bash
pip install -r requirements.txt
python run_demo.py --selftest
```

합성 영상으로 LED 점등부터 다중 프레임 확정까지 전체 흐름과 빠른 플랫필드, Hailo 입력 형식을 점검합니다.

### 2. Raspberry Pi 준비

```bash
sudo apt update && sudo apt install -y python3-picamera2 hailo-all
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -r requirements.txt
```

- `/boot/firmware/config.txt`에 `dtoverlay=pwm-2chan`을 넣고 재부팅합니다(채널0=GPIO18). `ls /sys/class/pwm`으로 chip 번호를 확인해 `preprocess.py`의 `PwmConfig.chip`에 넣으세요.
- PCIe Gen3 사용(`dtparam=pciex1_gen=3`)을 켜면 Hailo 처리량이 늘어납니다.
- `hailortcli fw-control identify`로 Hailo가 인식되는지 확인합니다.

### 3. 현장 캘리브레이션 (로봇 카메라로 촬영)

```bash
python offline_tools.py calibrate horizon
python offline_tools.py calibrate flatfield --images calib_raw/flat --out calib/flatfield_led.npy --preview
python offline_tools.py calibrate specular --images calib_raw/glare --out calib/specular_mask.npy --preview
```

- 먼저 `preprocess.py`의 `CameraConfig`를 체커보드 캘리브레이션 값으로 바꾸세요.
- 사진은 **LED 모드와 같은 설정**(노출 4ms, 게인 고정, AWB 고정, 샤프닝 0, 같은 노이즈 제거 모드, 같은 초점)으로 찍어야 합니다.

### 4. 학습 데이터 만들기 (PC)

```bash
python offline_tools.py dedup --images raw/session01/images --labels raw/session01/labels --out dedup/session01
python offline_tools.py build --images raw/train/images --labels raw/train/labels --out ds/train --variants 3 --cutouts cutouts
python offline_tools.py build --images raw/val/images --labels raw/val/labels --out ds/val --variants 0
yolo detect train model=yolo11n.pt data=data.yaml cfg=train_hyp.yaml
```

- 라벨은 원본 프레임(1280×720) 기준 YOLO 형식입니다. LED를 켜고 찍은 사진은 파일명에 `_led`를 넣어 주세요.
- train/val/test 분할은 **세션·위치 단위**로 `offline_tools.py build` 전에 끝내야 합니다.
- Copy-Paste용 물체 사진은 배경을 투명하게 뺀 RGBA PNG로 준비합니다. 파일명 규칙은 `<클래스번호>_<실제크기cm>_<이름>.png`입니다(예: `2_2.0_ring01.png`).

### 5. Hailo 모델 만들기 (PC, x86 Linux)

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=480,672 opset=11
python offline_tools.py calib-set --tiles ds/train/images --out calib_set --n 1024 --led-ratio 0.3
hailomz compile yolov8n --ckpt best.onnx --hw-arch hailo8l --calib-path calib_set/images --classes 12
```

- Hailo Dataflow Compiler / Model Zoo는 Hailo 개발자 사이트에서 받아 x86 Linux(또는 Docker)에서 실행합니다. Pi에서는 컴파일할 수 없습니다.
- 입력 크기 480×672는 Model Zoo 기본값(640×640)과 달라서, 모델 설정 yaml의 입력 크기를 바꿔야 할 수 있습니다. Model Zoo 버전에 따라 YOLO11 지원 여부와 명령이 다르니 설치한 버전의 문서를 확인하세요. 확실하게 하려면 YOLOv8n으로 시작하는 것을 권장합니다.
- NMS가 HEF에 포함되도록 컴파일해야 `HailoDetector`가 결과를 바로 읽을 수 있습니다.
- 컴파일 후에는 val 데이터로 **ONNX와 HEF의 작은 물체 Recall을 비교**해 INT8 손실을 확인하세요.

### 6. 실행

```bash
python run_demo.py --video sofa_led.mp4 --led on --weights best.pt --out demo_out
python run_demo.py --live --weights model.hef
```

- 첫 번째 줄은 PC에서 녹화 영상으로 검증할 때(Ultralytics), 두 번째 줄은 Pi에서 실시간으로 실행할 때 씁니다.
- `--live`에서는 50프레임마다 처리 시간 중앙값을 출력합니다. Pi 실측값으로 설계 문서의 연산 예산(약 30～45ms)을 확인하세요.

## ROS 2 노드에 연결하기 (예시)

```python
from preprocess import Config, OnboardPipeline
from rpi_hardware import HailoDetector, LedPwm, RpiCamera

cfg = Config()
pipe = OnboardPipeline(cfg, HailoDetector("model.hef"))
cam, led = RpiCamera(cfg), LedPwm(cfg)

def on_timer():                                  # 10Hz 타이머 콜백
    frame, meta, _ = cam.capture()
    res = pipe.step(frame, meta, robot_pose=lookup_pose_map_base_link())   # tf2로 구현
    led.set(res.led.led_on)                      # 다음 프레임부터 적용
    cam.set_mode(res.led.fixed_exposure)
    for track in res.confirmed:
        stop_and_notify(track.cls, track.pos)    # 정지, 알림, 대시보드 표시
```

전환 직후 프레임을 버리는 처리 등 전체 흐름은 `run_demo.py`의 `live()`를 참고하세요.

## 라즈베리 파이에서 주의할 점

1. **CPU를 나눠 씀:** SLAM·Nav2·ROS 2가 같은 4코어를 씁니다. 이 파이프라인은 별도 프로세스로 띄우고 코어를 고정하세요(예: `taskset -c 2,3 python ...`, 필요하면 `cv2.setNumThreads(2)`). 오프라인 STT(Whisper)는 탐색 중에 멈추세요.
2. **발열:** 약 85°C에서 클럭이 떨어집니다. 액티브 쿨러를 달고 30분 연속 구동 후 `vcgencmd measure_temp`, `vcgencmd get_throttled`로 확인하세요.
3. **카메라 설정:** Picamera2 기본값은 샤프닝이 켜져 있고 오토포커스입니다. `rpi_hardware.RpiCamera`가 이 둘을 끄고 고정합니다. 학습용 사진도 같은 설정으로 찍어야 합니다.
4. **ROS 2:** Picamera2와 Hailo 패키지는 Raspberry Pi OS용이라, ROS 2(Jazzy)는 Docker로 띄우거나 Ubuntu 24.04에서 카메라·Hailo 드라이버를 직접 설치해야 합니다.

## 확인이 필요한 부분 (Pi 실기에서만 확인 가능)

- `rpi_hardware.py`(`RpiCamera`, `LedPwm`, `HailoDetector`)는 Raspberry Pi 전용이라 PC에서 실행해 보지 못했습니다. 지원하지 않는 카메라 컨트롤은 건너뛰도록 만들었지만, libcamera·picamera2 버전에 따라 이름이 다를 수 있습니다.
- `HailoDetector`는 NMS가 포함된 HEF의 출력 형식(클래스별 `[y1, x1, y2, x2, score]`)을 가정합니다. 컴파일 방식이 다르면 `_parse`를 맞춰야 합니다.
- `preprocess.py` 맨 위 설정값의 모든 수치(LED 임계값 100/70, CLAHE 2.0, 초점 3.0 디옵터, 노이즈 제거 모드 등)는 가상 현장 기준 초기값입니다.

## 설계 문서와 달라진 점 (구현하면서 보완)

1. **LED를 끄는 판단:** LED를 켠 뒤에는 노출이 고정되어 노출×게인 값이 변하지 않습니다. 그래서 확인용 프레임(LED를 끈 1프레임)에서는 **고정 노출 상태의 ROI 평균 밝기**(`probe_off_mean`=70)로 판단합니다.
2. **증강 순서:** **Copy-Paste → 물리 열화** 순서입니다. 붙여넣은 물체도 노이즈·블러·비네팅을 똑같이 받아야 하기 때문입니다.
3. **바닥 위치 계산:** 지도 좌표는 **박스 하단 중앙(접지점)**으로 계산합니다.
4. **타일 병합:** IoU 대신 **IoMin**(교집합 / 작은 박스 면적)을 기준으로 병합합니다.
5. **4px 미만 물체:** 라벨만 지우면 배경으로 학습되므로 그 영역을 주변 픽셀로 메웁니다(inpaint).
