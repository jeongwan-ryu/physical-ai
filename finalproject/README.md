# Final Project: 손 제스처로 Gazebo TurtleBot3 조종

XIAO ESP32S3 Sense가 카메라 프레임에서 손 제스처를 판정하고(ESP-DL 사전학습 모델),
결과를 JSON 문자열로 보내면 PC의 제어 노드가 Gazebo TurtleBot3에 `/cmd_vel`을 내린다.

```
XIAO ESP32S3 (ESP-IDF)                        PC (ROS 2 Humble)
camera -> hand_detect -> gesture_cls (core 0)
camera -> JPEG (core 1)
   └─ USB Serial ──► serial_bridge ──► /gesture (String, JSON) ──► teleop_node ──► /cmd_vel ──► Gazebo TB3
                          └────────► /esp32_cam/image/compressed ──► dashboard
                                        /scan, /odom, /imu ──► teleop_node (충돌 안전)
```

## 폴더 구성

```
finalproject/
├── firmware/                  ESP-IDF v5.3 프로젝트 (target: esp32s3)
│   ├── sdkconfig.defaults     PSRAM(octal), 240MHz, USB-Serial/JTAG 콘솔
│   ├── partitions.csv
│   └── main/
│       ├── idf_component.yml  esp32-camera, hand_detect, hand_gesture_recognition
│       ├── camera_pins.h      XIAO ESP32S3 Sense 핀
│       └── main.cpp
└── ros2_ws/src/gesture_teleop/   ament_python 패키지
    ├── launch/sim.launch.py   TB3 월드(gzserver) + 상단 카메라 + 대시보드 + serial_bridge + teleop_node
    ├── models/overhead_cam.sdf  Gazebo 상단 고정 카메라 -> /overhead_cam/image_raw
    └── gesture_teleop/
        ├── teleop_node.py     /gesture -> /cmd_vel, 장애물 차단 / 충돌 감지
        ├── serial_bridge.py   Serial JSON -> /gesture, IMG 줄 -> /esp32_cam/image/compressed
        └── dashboard.py       웹 대시보드: Gazebo/ESP32 영상, 제스처/cmd_vel, /rosout 로그
```

## 대시보드 (http://localhost:8090)

- 왼쪽: Gazebo 상단 카메라 영상, ESP32 카메라 영상(MJPEG), 현재 제스처 / 신뢰도 / cmd_vel
- 오른쪽: 로그 (`/rosout` 전체 + `/gesture`, `/cmd_vel`은 값이 바뀔 때만)
- 상단 표시등: `/gesture`가 1초 넘게 안 오면 "ESP32 끊김"
- 8080, 8000은 다른 프로젝트가 사용 중이라 8090. 바꾸려면 `--ros-args -p port:=XXXX`

## 메시지 (`/gesture`, std_msgs/String)

```json
{"g":"like","s":0.97,"x":0.42,"seq":1234}
```
`g` 제스처, `s` 신뢰도, `x` 손 bbox 중심 x(0~1), `seq` 순번.
같은 시리얼에 `IMG:<base64 JPEG>` 줄과 ESP_LOG 로그도 섞여 나온다.

## 제스처 매핑

| 제스처 | 동작 |
|---|---|
| `five` ✋ | 즉시 정지, 충돌 잠금 해제 |
| `like` 👍 | 전진 |
| `dislike` 👎 | 후진 |
| `one` ☝️ | 좌회전 |
| `two` ✌️ | 우회전 |

판정 (`teleop_node`, ESP32 프레임마다)
- 동작 제스처가 `debounce`(2)번 나오면 반영. 사이에 낀 `no_gesture`/`no_hand`는 무시
  (모델이 one, no_gesture, one, ...처럼 번갈아 내놓음)
- 중립 프레임(`no_hand`, `no_gesture`, s < 0.7)이 `hold`(6)프레임 넘게 이어지면 정지
- `/gesture` 0.5초 무응답이면 정지. 노드 종료 시에도 정지 명령을 보낸다

충돌 안전
- `/scan` 전/후방 ±30° 안 `stop_dist`(0.2m) 이내면 그 방향 직진 차단 (WARN)
- 충돌 감지 시 정지 + 잠금 (ERROR). ✋ 또는 손을 내렸다 다시 해야 움직임
  - `/imu` 수평 가속도 > `bump_accel`(3 m/s²). 출발/정지 반동 때문에 명령 변경 후 0.5초는 무시
  - 직진 명령 중인데 `/odom` 속도가 명령의 25% 미만으로 `stuck_time`(0.7s)
- 명령이 0인데 계속 움직이면 WARN

## 실행

```bash
# 터미널 1: 전체 (Gazebo 서버, 대시보드 http://localhost:8090, serial_bridge, teleop_node)
cd ros2_ws && source /opt/ros/humble/setup.bash
colcon build --symlink-install && source install/setup.bash
source /usr/share/gazebo/setup.sh
ros2 launch gesture_teleop sim.launch.py            # gui:=true 면 Gazebo 창도, port:=/dev/ttyACM1 로 포트 변경

# 펌웨어 (터미널 1을 먼저 끌 것: 시리얼 포트 공유 불가)
source ~/.espressif/tools/activate_idf_v5.3.6.sh   # EIM 설치라 export.sh 대신 이걸 사용
cd firmware && idf.py set-target esp32s3 && idf.py build flash monitor

# 테스트
cd ros2_ws/src/gesture_teleop
PYTHONPATH=. python3 test/test_filter.py && PYTHONPATH=. python3 test/test_extract.py
```

`turtlebot3_drive`와 같이 켜면 `/cmd_vel`이 충돌한다.

## 성능 (XIAO ESP32S3, 손 없을 때)

7.4 fps. 손 검출이 프레임당 ~130ms로 병목(모델 1종뿐, sdkconfig는 이미 최적).
JPEG 인코딩은 코어 1로 분리했고, 손 추적 중엔 `DET_EVERY`(3)프레임마다만 검출한다.
30프레임마다 단계별 시간 로그: `7.4 fps | cam 0 det 134 cls 0 ms/frame | det 30/30 img 30/30`

## 진행 순서

1. teleop_node + `ros2 topic pub /gesture ...`로 Gazebo 동작 확인 (ESP32 없이)
2. ESP32 카메라 + ESP-DL 추론 → Serial JSON 확인
3. (현재) serial_bridge로 연결. micro-ROS(`micro_ros_espidf_component` humble)는 선택 과제
