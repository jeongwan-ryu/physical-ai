# Nooki SBC 시스템 디자인 (초안 v0.1, 2026-10-16)

근거: `PRD/PRD.md`, `DB/Nooki_chatbot/knowledge/nooki_prd.md`(하드웨어 보고서 10.14, 추론 백엔드 비교 10.15), `code/lost_item_preprocessing/`, `code/raspberrypi/benchmark.py`.

범위: **Raspberry Pi 5 위에서 도는 소프트웨어**. ESP32-S3 펌웨어는 컨테이너가 아니라 경계(토픽 계약)만 정의한다.

---

## 1. 원칙

1. **컨테이너 1개 = 담당자 1명 = 하드웨어 장치 묶음 1개.** 장치(`/dev/...`)를 두 컨테이너가 나눠 쓰지 않는다.
2. **"죽으면 안 되는 쪽"과 "죽어도 되는 쪽"을 나눈다.**
   - 장치 계층 (mcu, lidar, camera, hmi): 장치에 접근하고 가볍다. `restart: always`.
   - 소프트웨어 계층 (inference, nav, mission, dashboard, web): 장치 권한이 없다(inference만 `/dev/hailo0`). `mem_limit`을 두고 `restart: on-failure`.
   - 그래서 추론이 메모리 부족(OOM)으로 죽어도 카메라·주행·안전은 계속 돈다.
3. **컨테이너 사이 통신은 ROS 2 토픽/액션만.** 공유 파일·공유 코드 패키지 없음. 메시지는 **표준 타입만** 쓴다(`nooki_msgs` 같은 공용 패키지를 만들면 모든 이미지가 그걸 빌드해야 해서 병렬 작업이 깨진다).
4. **broker 컨테이너 없음.** ROS 2(DDS)는 노드끼리 직접 연결한다. 컨테이너가 늘어 discovery 트래픽이 문제 되면 그때 ROS 2 Discovery Server를 추가한다. MQTT는 쓰지 않는다(ROS 2와 중복).
5. 모든 ROS 컨테이너: `FROM ros:jazzy-ros-base`, `network_mode: host`, `ipc: host`, 같은 `ROS_DOMAIN_ID`. compose 파일 1개.
6. 영상은 로봇 밖으로 안 나간다. 외부로 열리는 포트는 대시보드(Wi-Fi, 집 안 LAN)뿐.
7. 늦으면 위험한 것(충돌·낙하·통신 끊김 정지)은 ESP32가 혼자 처리한다. Pi 쪽 컨테이너가 전부 죽어도 로봇은 0.5초 안에 멈춘다.

## 2. 전체 구조

![Nooki SBC 아키텍처](architecture.png)

원본: `architecture.svg` (수정 후 PNG 재생성: `google-chrome --headless=new --window-size=1440,1090 --force-device-scale-factor=2 --screenshot=$PWD/architecture.png file://$PWD/architecture.svg`)

## 3. 컨테이너별 명세

| 컨테이너 | 계층 | 하는 일 | 장치 / 권한 | 주요 패키지 | 레포 위치 |
|---|---|---|---|---|---|
| `mcu` | 장치 | ESP32 ↔ ROS 브리지 | `/dev/ttyACM0` | `microros/micro-ros-agent:jazzy` (코드 0줄) | 없음 (compose만) |
| `lidar` | 장치 | 2D 스캔 | `/dev/ttyUSB0` | `sllidar_ros2` (C1 지원) | `lidar/` (Dockerfile + launch 인자) |
| `camera` | 장치 | 영상 발행, 밝기 판단, LED 제어 | `/dev/video*`, `/dev/media*`, `/dev/dma_heap`, `/dev/gpiochip*`, `/run/udev:ro` | libcamera(rpi fork), picamera2 | `camera/` |
| `hmi` | 장치 | 음성 입력, 알림음, 시작 버튼 | `/dev/snd`, `/dev/gpiochip*` | whisper.cpp(tiny/base), `gpiod` | `hmi/` |
| `inference` | 소프트웨어 | 전처리·추론·후처리·지도 투영·검출 확정 | `/dev/hailo0`만 | HailoRT, ncnn, `lost_item_preprocessing` 코드 | `inference/` |
| `nav` | 소프트웨어 | URDF·TF, SLAM, Nav2, 커버리지 경로 | 없음 | `slam_toolbox`, `nav2_bringup`, `robot_state_publisher` | `nav/` |
| `mission` | 소프트웨어 | 탐색 시나리오 오케스트레이션 | 없음, 볼륨 `./data` | rclpy, `nav2_simple_commander` | `mission/` |
| `dashboard` | 소프트웨어 | 웹 ↔ ROS 브리지 | 포트 9090 | `rosbridge_server` (코드 0줄) | 없음 |
| `web` | 소프트웨어 | 대시보드 정적 페이지 | 포트 8080 | `nginx:alpine` + roslibjs | `web/` |

→ 실제로 코드를 쓰는 컨테이너는 **camera, inference, nav, hmi, mission, web** 6개.

### 3.1 camera → inference

```
[camera]  Picamera2(1280×720, 10fps) ─ 메타데이터(노출×게인) → 어두우면 LED on (LedController)
             │ /camera/image_raw, /camera/led_on
             ▼
[inference] Preprocessor.run(frame, led_on) → 타일들      ← preprocess.py
          → Hailo HEF 추론 (예비: NCNN FP16)              ← benchmark.py의 load_ncnn 재사용
          → LostItemPostprocessor.from_yolo/process         ← postprocess.py
          → 픽셀 → 바닥평면 → base_link → map (TF 조회)    ← 새로 작성 (카메라 내부·외부 파라미터 필요)
          → MapConfirmer.update(frame_id, obs)              ← postprocess.py
          → /nooki/found
```

- LED 판단은 카메라 메타데이터가 있는 camera에 둔다. inference는 `/camera/led_on`의 최신값만 쓴다(LED는 자주 안 바뀌니 프레임과 몇 ms 어긋나도 무방).
- 영상 전달 비용: 720p BGR 약 2.7MB × 10fps ≈ 27MB/s. `ipc: host` 공유메모리 전송이면 Pi 5에서 감당 가능. 실측해서 문제면 camera가 바닥 ROI만 잘라 보낸다.

## 4. 토픽 계약 (병렬 작업의 기준선)

여기 적힌 이름·타입·frame_id가 바뀌면 이 문서를 먼저 고친다.

| 토픽 / 액션 | 타입 | 발행 → 구독 | 비고 |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | nav → mcu | Jazzy Nav2 기본(unstamped)과 맞춤. 최대 0.2 m/s |
| `/odom` | `nav_msgs/Odometry` | mcu → nav | frame `odom`→`base_link`. TF는 nav가 이 토픽으로 브로드캐스트 |
| `/imu` | `sensor_msgs/Imu` | mcu → nav | frame `imu_link` |
| `/mcu/safety` | `std_msgs/UInt8` | mcu → mission, web | 비트: 0 범퍼L, 1 범퍼R, 2~5 cliff, 7 비상정지 중 |
| `/scan` | `sensor_msgs/LaserScan` | lidar → nav | frame `laser` |
| `/camera/image_raw` | `sensor_msgs/Image` | camera → inference | bgr8, 1280×720, frame `camera` |
| `/camera/led_on` | `std_msgs/Bool` (transient_local) | camera → inference | 바뀔 때만 발행 |
| `/map` | `nav_msgs/OccupancyGrid` | nav → web, mission | slam_toolbox |
| `/nooki/coverage_path` | `nav_msgs/Path` | nav → mission | 지그재그 웨이포인트, frame `map` |
| `follow_waypoints` (action) | `nav2_msgs/FollowWaypoints` | mission → nav | `stop_on_failure: false` |
| `/nooki/target` | `std_msgs/String` | hmi, web → mission, inference | 클래스 이름 (`airpods`, `ring`, `earring`, `key`, `remote`) |
| `/nooki/cmd` | `std_msgs/String` | hmi, web → mission | `start` / `stop` |
| `/nooki/state` | `std_msgs/String` (transient_local) | mission → 전체 | `IDLE` `SEARCHING` `FOUND` `FAILED` |
| `/nooki/detections` | `vision_msgs/Detection2DArray` | inference → web | 프레임별 원시 검출 (디버그) |
| `/nooki/found` | `vision_msgs/Detection3DArray` | inference → mission | frame `map`, 확정된 것만 |
| `/nooki/found_image` | `sensor_msgs/CompressedImage` | inference → mission, web | 확정 순간 박스 그린 JPEG |
| `/nooki/debug_image` | `sensor_msgs/CompressedImage` | inference → web | 2 Hz 이하, `SEARCHING` 중만 |
| `/nooki/stt_enable` | `std_msgs/Bool` | mission → hmi | 탐색 중 STT 끔 (CPU 확보) |

TF 트리: `map → odom → base_link → {laser, camera, imu_link}`
- `map→odom`: slam_toolbox / `odom→base_link`: nav (/odom 기반) / 나머지 고정: nav의 URDF.
- 카메라 높이·각도는 URDF 한 곳에만 둔다. inference는 TF로 읽어간다(값 복사 금지).

## 5. mission 상태머신

```
IDLE ──(target 설정 & cmd=start)──▶ SEARCHING
  SEARCHING 진입: stt_enable=false, coverage_path → follow_waypoints
  SEARCHING ──/nooki/found──▶ FOUND   : 웨이포인트 취소, 정지, 알림음, 사진+위치 ./data 저장
  SEARCHING ──3분 경과 / 경로 끝──▶ FAILED
  SEARCHING ──cmd=stop──▶ IDLE
FOUND / FAILED ──cmd=start 또는 stop──▶ IDLE (stt_enable=true)
```

inference가 탐색 중 죽으면: 재시작될 때까지 검출만 멈추고 주행은 계속된다. 3분 안에 못 찾으면 FAILED로 끝난다.

## 6. compose 골격

```yaml
x-ros: &ros
  network_mode: host
  ipc: host
  environment: { ROS_DOMAIN_ID: "42" }
x-hw: &hw   { <<: *ros, restart: always }                         # 장치 계층
x-sw: &sw   { <<: *ros, restart: on-failure, mem_limit: 1g }      # 소프트웨어 계층

services:
  mcu:       { <<: *hw, image: microros/micro-ros-agent:jazzy,
               command: serial --dev /dev/ttyACM0 -b 921600, devices: [/dev/ttyACM0] }
  lidar:     { <<: *hw, build: ./lidar, devices: [/dev/ttyUSB0] }
  camera:    { <<: *hw, build: ./camera, privileged: true, volumes: [/run/udev:/run/udev:ro] }
  hmi:       { <<: *hw, build: ./hmi, devices: [/dev/snd, /dev/gpiochip0] }
  inference: { <<: *sw, build: ./inference, devices: [/dev/hailo0], mem_limit: 2g }
  nav:       { <<: *sw, build: ./nav }
  mission:   { <<: *sw, build: ./mission, volumes: [./data:/data], mem_limit: 256m }
  dashboard: { <<: *sw, image: ros:jazzy-ros-base,  # + apt ros-jazzy-rosbridge-server
               command: ros2 launch rosbridge_server rosbridge_websocket_launch.xml, mem_limit: 256m }
  web:       { image: nginx:alpine, restart: on-failure, mem_limit: 64m,
               ports: ["8080:80"], volumes: [./web:/usr/share/nginx/html:ro] }
```

- `mem_limit` 값은 추정치. 실측 후 조정한다(Pi 5 8GB).
- `camera`의 `privileged: true`는 초안용. 장치 목록이 확정되면 `devices:`로 좁힌다.

## 7. 하드웨어 없이 개발하는 법

| 컨테이너 | 대체 입력 |
|---|---|
| nav | 실기 rosbag(`/scan /odom /imu /tf_static`) 재생. 실기 전엔 Nav2 기본 TurtleBot3 Gazebo 시뮬 |
| camera | x86에서 `--source video.mp4`로 같은 토픽 발행 |
| inference | rosbag의 `/camera/image_raw` 재생 + NCNN 백엔드. TF는 `static_transform_publisher`로 가짜 `map→camera` |
| hmi | 노트북 마이크 (`/dev/snd`), 버튼은 키보드 엔터로 대체 |
| mission | `ros2 topic pub /nooki/found ...` 수동 발행, Nav2는 시뮬 |
| web | 노트북에서 `dashboard` + `web`만 띄우고 `ros2 topic pub`로 가짜 데이터 |
| ESP32 펌웨어 | micro-ROS agent에 붙여 `ros2 topic echo /odom`으로 확인 |

**rosbag을 가장 먼저 1개 만든다** (실기에서 방 하나 수동 주행, `/camera/image_raw` 포함). 이후 모든 담당자가 같은 데이터로 작업한다. camera를 분리한 덕분에 inference는 실기 없이 rosbag만으로 개발된다.

## 8. 작업 분할과 완료 기준

| 담당 | 바로 시작 가능한 것 | 완료 기준 |
|---|---|---|
| A: camera + inference | Ubuntu 24.04 컨테이너에서 Picamera2 / HailoRT 동작 확인(아래 리스크 1) | Pi 5에서 `/nooki/found`가 지도 좌표 ±10cm, 10fps(Hailo) 또는 5fps(NCNN). inference를 `docker kill`해도 camera·주행 유지 |
| B: nav | URDF 작성, slam_toolbox·Nav2 파라미터, 커버리지 경로 노드 | 방 하나 지도 작성 + 웨이포인트 주행 완주, 충돌 0 |
| C: 펌웨어 + mcu | micro-ROS 펌웨어, 토픽 4개 | `/cmd_vel` 0.2 m/s 추종, 범퍼·cliff 즉시 정지, 통신 0.5초 끊김 정지 |
| D: hmi | whisper.cpp 컨테이너, 한국어 키워드 → 클래스 매핑 | "에어팟 찾아줘" → `/nooki/target=airpods` 2초 내 |
| E: mission + web | 상태머신, 대시보드 페이지 | 대시보드에서 시작 → 검출 사진·지도 위치 표시까지 end-to-end |

A는 인원이 늘면 camera / inference로 나눈다(경계는 `/camera/image_raw` 토픽 하나).

순서: **M0** compose 골격 + 리스크 1 확인 → **M1** mcu+lidar+nav로 원격조종·지도 → **M2** camera+inference 벤치 검출 → **M3** mission으로 전체 연결, 3분 KPI 측정.

## 9. 리스크 / 미결정

1. **(최우선) ROS Jazzy 이미지는 Ubuntu 24.04, Picamera2·HailoRT는 Raspberry Pi OS(Debian 12) 기준 배포.** camera(rpi libcamera 소스 빌드)와 inference(HailoRT arm64 설치)가 되는지 M0에서 확인. HailoRT 버전은 호스트 PCIe 드라이버 버전과 반드시 같아야 함. 안 되면 대안: 해당 컨테이너만 Bookworm 베이스 + ROS 없이 돌리고 rosbridge(WebSocket)로 토픽 발행.
2. CPU 4코어에 SLAM + Nav2 + (NCNN 예비 시) 추론이 겹침. 측정 후 필요하면 compose `cpuset`으로 inference를 코어 3개에 묶는다.
3. 커버리지 경로: SLAM 지도가 탐색 시작 시점엔 비어 있음. MVP는 시작 위치 기준 고정 사각형(예: 4×4m) 지그재그 + Nav2 장애물 회피로 막힌 점은 건너뜀. 방 모양에 맞춘 경로는 지도 완성 후 2차.
4. 픽셀→지도 투영은 카메라 캘리브레이션(내부 파라미터 + 장착 높이·각도) 필요. 바닥 평면 가정이라 가구 밑 경사·러그에서 오차.
5. `odom→base_link`를 휠 오도메트리만으로 시작. 바퀴 미끄러짐이 SLAM을 망치면 nav에 `robot_localization`(EKF, IMU 융합) 추가.
6. FastDDS 공유메모리는 `ipc: host` + 같은 UID 전제. 컨테이너 간 토픽이 안 보이거나 영상이 느리면 우선 의심.
7. PRD 반영 필요: ESP32-S3 추가, NCNN 예비 시 목표 5fps.
