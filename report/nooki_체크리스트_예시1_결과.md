# 🧪 분실물 검출 무인이동체 — 데이터 전처리 설계안 (HOME-01, Raspberry Pi 5 버전)

> 이 문서는 **임의로 채운 가상 현장 데이터(HOME-01: 30평대 아파트 거실, 밝은 회색 유광 타일, 패브릭 소파 밑, 고양이 1마리)** 를 근거로 작성했습니다.
> 파라미터와 연산 시간은 모두 그 데이터에서 출발하는 **초기값과 추정치**이므로 실제 장비에서 측정해 조정해야 합니다.
> **하드웨어:** Raspberry Pi 5 8GB + AI HAT+ 13 TOPS(Hailo-8L) · Camera Module 3 Wide(수평 102°) · RP1 하드웨어 PWM LED

---

## 0. 설계 결론부터 (TL;DR)

1. **가장 큰 문제는 해상도이고, 필터는 그다음입니다.** 1280×720 화면을 960으로 줄이면 반지(10px)는 약 7px, 귀걸이(4px)는 약 3px가 되어 어떤 필터로도 살릴 수 없습니다. 그래서 **쓸모없는 위쪽 화면을 잘라내고(ROI 크롭), 남은 부분을 원본 해상도 그대로 타일로 나눠 추론**하는 것이 출발점입니다.
2. **소프트웨어로 보정하기 전에 촬영 단계(LED·노출·ISP)부터 고쳐야 합니다.** 하얗게 날아간 픽셀은 CLAHE로 되살릴 수 없습니다. GPU가 없는 라즈베리 파이에서는 노이즈 제거처럼 무거운 일을 **카메라 ISP에 맡기는 것**이 CPU를 아끼는 가장 좋은 방법입니다. 반대로 ISP 기본 샤프닝은 꺼야 합니다.
3. **유광 타일에 비친 LED 반사는 화면 속 같은 자리에 생깁니다.** 카메라, LED, 바닥이 고정된 기하 관계라서 반사 위치도 고정됩니다. 그래서 미리 그 자리를 **마스크로 만들어 두고 후처리에서 걸러낼 수 있습니다.**
4. **"프레임 평균 밝기로 LED 점등 판단"은 그대로 쓰면 동작하지 않습니다.** 자동 노출이 평균 밝기를 늘 중간값 근처로 맞추기 때문입니다. 대신 Picamera2 메타데이터의 **노출 시간 × 아날로그 게인**으로 판단해야 합니다.
5. **학습에 쓰는 전처리와 로봇에서 쓰는 전처리는 같은 코드여야 합니다.** 이제 학습(PC)과 추론(Pi)이 모두 CPU OpenCV라 결과가 완전히 같습니다. 노이즈·블러·조명 같은 "물리적 열화" 증강은 전처리 **앞**에 넣어야 합니다.
6. **연산 배치: ISP → CPU → NPU.** 노이즈 제거는 ISP(0ms), 크롭·플랫필드·CLAHE·타일링·후처리는 CPU(약 12～20ms), YOLO는 Hailo NPU(약 15～25ms)에서 처리합니다. CPU는 SLAM·Nav2와 나눠 쓰므로 전처리는 가벼운 연산만 남깁니다.

---

## 1. 파이프라인 (A) — 학습 데이터 구축용 오프라인 전처리 + 증강 (PC)

### A-1. 데이터 정제 (한 번만 실행)

| 순서 | 단계 | 기법 | 파라미터(초기값) | 적용 이유 | 연산 비용 |
|---|---|---|---|---|---|
| 1 | 수집 소스 통일 | 로봇 카메라 원본 프레임만 사용 (폰 사진은 보조로만, 전체의 20% 이하) | 로봇과 **같은 Picamera2 설정**(노출·LED·샤프닝 0·노이즈 제거 모드·고정 초점), PNG 또는 JPEG 품질 95 이상 | 폰은 HDR·샤프닝 같은 자체 처리를 거쳐서 로봇 화면과 분포가 다릅니다. 압축 손상은 10px 물체에 치명적입니다 | – |
| 2 | 개인정보 처리 | 사람 검출 → 해당 프레임 폐기 또는 영역 블러 | 사람 검출 conf 0.3 | 7-3번(가족 발 노출) 대응입니다. 저장하기 전에 처리해야 합니다 | 낮음 |
| 3 | 프레임 샘플링 | 영상(45초 × 4개)에서 pHash로 중복 제거 | 해밍 거리 6 이하면 중복으로 보고 제거, 대략 0.3초마다 1장 | 연속 프레임은 거의 같은 이미지라서 과적합과 검증 누수를 일으킵니다 | 낮음 |
| 4 | 데이터 분할 | **세션·위치 단위**로 나누기 (프레임 단위 금지) | train/val/test = 70/15/15, 조건별(낮 개방 / 밤 25 lux / 소파 밑 LED)로 층화 | 같은 장면이 train과 val에 섞이면 성능이 부풀려집니다 | – |
| 5 | 라벨링 | 대상 6클래스 + **방해 물체 클래스**(사료, 알약, 병뚜껑, 머리끈, 플러그, LED 반사광) | 4px 미만 물체는 ignore 영역으로 처리, 박스는 물체에 딱 맞게 | 방해 물체를 배경으로 두는 것보다 따로 클래스를 주는 편이 오검출을 훨씬 확실하게 줄입니다(5-2: 방당 4개 이상) | 사람 작업 |

### A-2. 증강 사본 생성과 학습 (이 순서대로)

| 순서 | 단계 | 기법 | 파라미터(초기값) | 적용 이유 |
|---|---|---|---|---|
| 1 | **Copy-Paste** | SAM으로 물품을 분리한 뒤 바닥 배경에 붙여넣기. **바닥 평면 기하에 맞춰 화면 높이(행 y)별로 크기를 정함**, 그림자 합성, 물품 자체 회전 0～360° | 이미지당 1～4개, 30%는 가구 다리·멀티탭 옆에 붙여 부분 가림 | 작은 물체 데이터를 늘리는 가장 효과적인 방법입니다. 붙여넣은 물체도 다음 단계의 열화를 똑같이 받도록 먼저 붙입니다 |
| 2 | **물리 열화 증강** (원본 화면 단계) | ① 조명: 밝기 ±30%, 색온도 2700～6500K ② 방사형 비네팅 ③ 센서 노이즈(Poisson-Gaussian) ④ 선형 모션블러 3～7px(수평 가중) ⑤ 렌즈 털·뿌연 막 ⑥ **가짜 LED 반사점** | 각 p=0.3～0.5 | 로봇에서는 이런 열화가 전처리 **전에** 일어납니다. 그러니 학습에서도 전처리 전에 넣어야 같은 분포가 됩니다 |
| 3 | 결정적 전처리 (B와 **같은 코드**) | ROI 크롭 → 플랫필드 보정 → CLAHE | (B)와 동일 | 학습과 추론의 전처리를 똑같이 맞춥니다 |
| 4 | 타일링 | 672×480 타일, overlap 64px | 타일 경계에서 잘린 박스는 60% 이상 남은 것만 유지 | 추론과 같은 입력 크기와 스케일을 씁니다 |
| 5 | 기하 증강 (학습 중, 가벼운 것만) | 좌우 반전, scale ±20%, translate 0.1, mosaic p=0.5(마지막 10 epoch는 끔) | **상하 반전·큰 회전 금지** | 카메라 높이와 각도가 고정이라 원근 구조가 일정합니다 |
| 6 | 색 증강 (학습 중, 약하게) | HSV h ±0.015, s ±0.3, v ±0.3 | – | 흰 에어팟, 금 귀걸이, 갈색 사료를 구분하는 색 단서를 지키려면 약하게 걸어야 합니다 |

### A-3. Hailo 모델 변환 (PC, x86 Linux)

| 순서 | 단계 | 내용 | 적용 이유 |
|---|---|---|---|
| 1 | ONNX 내보내기 | `yolo export format=onnx imgsz=480,672` | Hailo NPU는 ONNX를 직접 실행하지 못하고, 컴파일된 HEF만 실행합니다 |
| 2 | **양자화 보정 세트** | (B) 전처리를 마친 타일 1,024장, LED 모드 30% 이상, 반사광 포함 장면 10% 이상, uint8 RGB | Hailo DFC가 INT8 범위를 이 이미지로 잡습니다. 어두운 LED 모드 영상이 빠지면 소파 밑 정확도가 크게 떨어집니다 |
| 3 | HEF 컴파일 | Hailo Dataflow Compiler / Model Zoo, `--hw-arch hailo8l`, NMS 포함 | Pi에서는 컴파일할 수 없어 PC에서 만듭니다 |
| 4 | 정확도 비교 | val 세트로 ONNX vs HEF의 4～16px 물체 Recall 비교 | INT8 변환 손실은 작은 물체에서 가장 크게 나타납니다 |

---

## 2. 파이프라인 (B) — 로봇 온보드 실시간 추론 전처리 (Raspberry Pi 5)

### B-0. 촬영·ISP 설정 (가장 먼저, CPU 비용 0)

| 항목 | 설정 (Picamera2 컨트롤) | 근거 |
|---|---|---|
| LED PWM | RP1 하드웨어 PWM(`dtoverlay=pwm-2chan`) **20kHz 이상**, 또는 정전류 드라이버 | PWM 주파수가 낮으면 롤링셔터 카메라에 가로 줄무늬가 생깁니다(1-7번과 같은 현상) |
| LED 확산판·각도 | 확산판 부착, LED를 카메라보다 2～3cm 위·바깥쪽으로 배치 | 20cm 이내 과노출(6-3번)과 점 모양 반사를 줄입니다 |
| 교차 편광 (실험 권장) | LED에 선형 편광 필름, 렌즈에 그와 직교하는 편광 필터 | 유광 타일의 정반사를 크게 줄입니다. 대신 빛이 약 2스톱 줄어 LED 출력을 올려야 합니다 |
| 개방 공간 노출 | `AeEnable=True`, `AeFlickerMode=Manual`, `AeFlickerPeriod=8333`µs | 60Hz 전원의 천장 LED 깜빡임(120Hz) 대응입니다 |
| LED 모드 노출 | `AeEnable=False`, `ExposureTime=4000`µs, `AnalogueGain=4` 이하, `AwbEnable=False`(직전 ColourGains로 고정) | 0.5rad/s로 회전하면 화면에서 약 260px/s로 움직입니다. 8ms면 2.1px, 4ms면 1.0px가 번지는데, 10px짜리 반지에 2px 번짐은 치명적입니다. 노출·화이트밸런스를 고정해야 플랫필드 보정도 정확해집니다 |
| **ISP 노이즈 제거** | `NoiseReductionMode`: HighQuality / Fast / Minimal 비교 후 선택 | 소파 밑 2 lux 노이즈를 CPU 대신 ISP 하드웨어가 처리합니다. 너무 강하면 4px 귀걸이가 뭉개지므로 실측으로 고릅니다 |
| **ISP 샤프닝 끔** | `Sharpness=0` (기본값 1.0) | 기본 샤프닝은 타일 줄눈·고양이 털을 강조해 작은 오검출을 늘립니다 |
| **초점 고정** | `AfMode=Manual`, `LensPosition≈3.0`(약 33cm) | Camera Module 3는 오토포커스라 주행 중 초점이 흔들리면 블러 프레임이 늘어납니다 |
| HDR | 끔 | 프레임마다 톤이 달라져 학습·추론 분포가 어긋납니다 |
| LED 점등 판단 | **밝기 지수 = 노출(ms) × 아날로그 게인**(메타데이터). 100 초과가 5프레임 이어지면 켬. 2초마다 LED를 1프레임 끄고, 고정 노출 상태의 ROI 평균 밝기 ≥ 70이면 끔. 확인용 프레임은 검출에 쓰지 않음 | 자동 노출이 평균 밝기를 맞춰버리므로 픽셀 평균으로는 판단할 수 없습니다. 임계값은 가정값이라 실측 보정이 필요합니다 |
| 전환 직후 프레임 | 노출 모드 전환 후 3프레임, LED 전환 후 1프레임 버림 | libcamera 컨트롤은 몇 프레임 뒤에 적용되고, 롤링셔터는 LED 전환 중간에 찍힌 프레임을 반쪽만 밝게 만듭니다 |

### B-1. 프레임 처리 순서

| 순서 | 단계 | 기법 | 파라미터(초기값) | 적용 이유 | 예상 시간 (Pi 5) | 처리 위치 |
|---|---|---|---|---|---|---|
| 1 | 품질 게이트 | 바닥 영역을 1/2로 줄인 회색 영상의 Laplacian 분산 + 포화 비율(OpenCV 연산) | 분산이 **최근 30프레임 중앙값 × 0.6 미만**이면 폐기. 반사 마스크 밖 포화 5% 초과면 LED duty 하향 | 질감 없는 타일에서는 고정 문턱값이 정상 프레임까지 버리므로 상대값을 씁니다 | 약 2～4ms | CPU |
| 2 | ROI 크롭 | 위쪽 240행 제거 | 1280×480 | 지평선(약 221행) 위는 가구 밑면과 벽입니다. ROI 맨 윗줄은 약 1.5m 앞 바닥이라 그보다 먼 바닥만 잘립니다 | 약 0 | CPU (복사 없음) |
| 3 | 플랫필드 보정 | 미리 측정한 게인 맵을 **감마 영역 게인(gain^(1/2.2))으로 변환해 두고 `cv2.multiply` 한 번** | LED 모드 전용, 게인 상한 2.5배 | 바닥이 평면이고 LED·카메라가 고정이라 **조명 분포가 늘 같은 모양**입니다. 프레임마다 선형 변환을 두 번 하면 Pi CPU에서 수십 ms가 걸려 미리 변환합니다 | 약 2～3ms | CPU |
| 4 | CLAHE | LAB 색공간의 L 채널에만 적용 | clipLimit **2.0** (1.5/2.0/3.0 비교), tileGrid **(8, 3)** | 흰 에어팟과 밝은 회색 타일처럼 대비가 낮은 경우의 핵심 대응입니다. 색 단서는 그대로 둡니다 | 약 6～10ms | CPU |
| 5 | 타일링 | 672×480 타일 2장(overlap 64px), **uint8 RGB** | 정규화(/255)는 HEF 안에서 처리 | 원본 해상도를 그대로 유지합니다 | 약 1ms | CPU |
| 6 | 추론 | YOLO11n(또는 YOLOv8n) HEF INT8 | 타일 2장 순차 입력 | – | **약 15～25ms(추정)** | Hailo NPU |
| 7 | 후처리 | 타일 병합(IoMin) → **반사 마스크 영역은 conf +0.2를 더 요구** → 최소 크기 필터 → 다중 프레임 확정 | 지도 좌표 반경 5cm 안에서 **5프레임 중 3프레임 이상** 검출. conf: 작은 클래스 0.35, 큰 클래스 0.5 | 로봇이 움직이면 실제 물체는 반사 마스크를 지나가지만 반사광은 같은 자리에 있습니다. 그래서 다중 프레임 검증이 반사 오검출을 걸러냅니다 | 약 1～2ms | CPU |
| | **합계** | | | | **약 30～45ms** (예산 100ms) | |

- CPU는 SLAM·Nav2·ROS 2와 함께 쓰므로, 이 파이프라인은 **별도 프로세스로 띄우고 코어를 고정**하세요(예: `taskset -c 2,3`, `cv2.setNumThreads(2)`).
- 오프라인 STT(Whisper)는 CPU를 크게 쓰므로 **탐색 중에는 멈추고** 물품 입력 단계에서만 돌립니다.
- Hailo 추론과 다음 프레임의 CPU 전처리를 겹쳐 실행(파이프라이닝)하면 지연을 더 줄일 수 있습니다.
- 위 시간은 모두 추정치입니다. Pi에서 단계별 `time.perf_counter()`와 `hailortcli benchmark`로 직접 재고, 30분 연속 구동 후 발열 스로틀링(`vcgencmd get_throttled`)이 없는지도 확인해야 합니다.

### B-2. 이 순서여야 하는 이유

| 순서 관계 | 반대로 하면 생기는 문제 |
|---|---|
| 촬영·ISP 설정 → 나머지 전부 | 클리핑이나 블러로 잃은 정보, ISP 샤프닝으로 강조된 노이즈는 이후 단계에서 되돌릴 수 없습니다 |
| ISP 노이즈 제거 → 플랫필드 | ISP는 센서 원본에 가까운 단계에서 노이즈를 지워 가장 효과적이고 CPU를 쓰지 않습니다. 플랫필드 뒤에 지우면 가장자리에서 2.5배로 커진 노이즈를 CPU로 지워야 합니다 |
| 품질 게이트 → 보정 | CLAHE를 먼저 걸면 Laplacian 분산이 인위적으로 커져서 흐린 프레임이 게이트를 통과합니다. 버릴 프레임에 CPU를 쓰는 낭비도 생깁니다 |
| 크롭 → CPU 연산 | 이후 모든 CPU 단계에서 처리할 픽셀이 33% 줄어듭니다 |
| 플랫필드 → CLAHE | 플랫필드는 "빛 × 반사율"이라는 곱셈 모델에 기반한 보정입니다. 비선형인 CLAHE를 먼저 걸면 이 모델이 깨집니다 |
| CLAHE(전체 프레임) → 타일링 | 타일마다 CLAHE를 따로 걸면 두 타일의 대비가 달라집니다. 그러면 겹치는 영역의 같은 물체가 서로 다르게 보입니다 |
| (학습) Copy-Paste → 물리 열화 증강 → 결정적 전처리 | 로봇에서는 열화가 먼저 일어나고 전처리가 나중입니다. 순서를 바꾸면 로봇에서는 나오지 않는 이미지로 학습하게 됩니다 |

---

## 3. 이 현장에서 하면 안 되는 전처리

| 기법 | 이유 |
|---|---|
| 전체 프레임을 640/960으로 리사이즈 | 반지 10px가 5～7px, 귀걸이 4px가 2～3px로 줄어서 물체가 사라집니다 |
| 샤프닝·언샤프 마스크·엣지 강화 (**ISP 기본 샤프닝 포함**) | 타일 줄눈, 고양이 털, 노이즈가 강해져서 작은 오검출이 폭증합니다(2-3, 2-5번) |
| CPU 공간 필터 (Median 5×5, 강한 Bilateral, NLM) | 반지는 얇은 고리 모양이라 median 한 번에 거의 사라집니다. GPU가 없는 Pi에서 NLM은 수백 ms가 걸립니다. 노이즈는 ISP에서 처리합니다 |
| 전역 히스토그램 평활화 | LED 근거리 과노출과 반사점이 화면 전체 분포를 왜곡합니다 |
| 추론 전 전체 프레임 왜곡 보정(Undistort) | 보간으로 작은 물체가 흐려지고 CPU 시간도 듭니다. 왜곡된 영상 그대로 학습·추론하고, 검출 좌표만 `undistortPoints`로 보정합니다 |
| 딥러닝 초해상화(SR) | 없는 디테일을 지어내서 오검출이 늘고, Hailo-8L 연산 예산도 넘깁니다 |
| 배경 차분 / 프레임 차분 | 카메라가 움직이므로 쓸 수 없습니다 |
| 흑백 변환 | 흰 에어팟과 갈색 사료, 금 귀걸이를 구분하는 색 단서를 잃습니다 |
| 조건부 전처리를 추론에서만 적용 | 학습-추론 불일치의 대표적인 원인입니다. 결정적 전처리는 하나의 모듈로 만들어 학습과 추론이 같이 import해야 합니다 |
| 카메라 HDR, 연속 오토포커스 | 프레임마다 톤·초점이 달라져 분포가 흔들리고 블러 프레임이 늘어납니다 |
| float NCHW 배치를 Hailo에 입력 | Hailo HEF는 uint8 NHWC 입력에 정규화가 내장되어 있습니다. /255를 한 번 더 하면 입력이 거의 검게 들어갑니다 |
| 상하 반전, 큰 각도 회전, 강한 Hue/Saturation 증강 | 고정된 카메라 원근 구조와 색 단서를 깨뜨립니다 |

---

## 4. 검증 실험 설계 (Ablation)

**테스트 세트 (조건별로 따로 평가)**
- T1 낮 개방 바닥 / T2 밤 무드등 25 lux / T3 소파 밑 LED
- T4 **대상 물품 없는 빈 방 주행 영상 10분** (분당 오검출 측정용)

**지표**
- 크기 구간별 Recall과 mAP50: **4～8px / 8～16px / 16～32px / 32px 이상**
- 분당 오검출(T4): 다중 프레임 확정 전과 후를 각각 측정
- 끝단 지표: 조건별 20회 탐색 시 성공률, 검출까지 걸린 시간, 프레임당 지연 시간(Pi 실측)
- **ONNX(FP32) vs HEF(INT8)** 구간별 Recall 차이

**실험 순서 (한 번에 하나씩 추가, 시드 고정, 각 3회 반복)**
1. 기준선: 전체 프레임 960 리사이즈, 전처리 없음, ISP 기본값(샤프닝 켜짐)
2. \+ ROI 크롭과 원본 해상도 타일링 → *가장 큰 개선이 예상되는 단계*
3. \+ 촬영 설정(LED 모드 노출 고정, 깜빡임 보정, 확산판, 초점 고정)
4. \+ ISP 샤프닝 끔, 노이즈 제거 모드 비교(HighQuality / Fast / Minimal)
5. \+ 플랫필드 보정
6. \+ CLAHE (clipLimit 1.5 / 2.0 / 3.0 비교)
7. \+ Copy-Paste와 물리 열화 증강
8. \+ 방해 물체 클래스
9. \+ 반사 마스크 후처리와 다중 프레임 확정
10. HEF 변환 후 같은 실험을 Pi에서 반복 (INT8 손실, 실제 지연 시간 확인)

**목표 예시:** 8～16px 구간 Recall 0.85 이상, 확정 후 오검출 탐색 1회당 0.2건 이하, Pi 지연 시간 50ms 이하.

---

## 5. 현장에서 추가로 측정할 것

1. **노출 × 게인 지수의 실측 대응표**: 2, 25, 60, 180 lux 각각에서 Picamera2 메타데이터로 측정해 LED 점등 임계값 100/70을 확정합니다.
2. **플랫필드 촬영**: 어두운 곳에서 LED를 켜고 무광 회색 시트를 PWM 단계별로 찍습니다.
3. **반사 마스크 촬영**: 유광 타일 위에서 LED를 켜고 찍어, 반사점 위치와 크기를 기록합니다.
4. **체커보드 캘리브레이션**: 렌즈 내부 파라미터와 왜곡 계수, 실제 지평선 행 위치를 구합니다.
5. **물체 픽셀 크기 실측**: 로봇 카메라로 직접 찍어 거리별 크기를 확인합니다. 귀걸이를 10px 이상으로 보려면 약 40cm 이내여야 하므로 **Nav2 커버리지 경로의 간격**을 정하는 데 씁니다.
6. **ISP 노이즈 제거 모드와 초점 위치 비교**: 4～10px 물체가 가장 덜 뭉개지는 모드, 20cm～1.5m 바닥이 가장 선명한 `LensPosition`을 찾습니다.
7. **실제 주행 속도에서 블러 측정**: 최대 회전 속도와 롤링셔터 영향을 확인합니다.
8. **교차 편광 A/B 비교**: 유광 타일과 은반지를 편광 유무로 각각 찍어 비교합니다.
9. **Pi 연산 시간·발열**: 단계별 CPU 시간, Hailo 추론 시간, SLAM·Nav2를 같이 돌릴 때의 프레임 시간, 30분 연속 구동 후 온도.
10. **다른 바닥 2곳 이상**: 마루와 장판 현장을 추가합니다.

---

## 6. 온보드 전처리 의사코드 (Picamera2 + OpenCV + Hailo)

```python
import cv2, numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo
from libcamera import controls

# ---- 사전 캘리브레이션 결과 (현장 측정으로 생성) ----
ROI_TOP   = 240                                            # 지평선(약 221행) 아래
FF_SRGB   = np.repeat(np.load("flatfield_led.npy")[..., None], 3, 2) ** (1 / 2.2)  # 감마 영역 게인
SPEC_MASK = np.load("specular_mask.npy")                   # (480,1280) bool, 15px 팽창
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 3))
hailo = Hailo("yolo11n_lostfound.hef")                     # 입력 480×672, NMS 포함

# ---- ① 촬영·ISP 설정 ----
cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},        # 메모리상 BGR → OpenCV 그대로
    controls={"FrameDurationLimits": (100000, 100000)}))   # 10fps
cam.start()
cam.set_controls({"Sharpness": 0.0,                        # ISP 샤프닝 끔
                  "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
                  "AfMode": controls.AfModeEnum.Manual, "LensPosition": 3.0})

def set_led_mode(on, colour_gains):
    if on:   # 노출·게인·AWB 고정
        cam.set_controls({"AeEnable": False, "ExposureTime": 4000, "AnalogueGain": 4.0,
                          "AwbEnable": False, "ColourGains": colour_gains})
    else:    # AE + 60Hz 깜빡임 보정
        cam.set_controls({"AeEnable": True, "AwbEnable": True,
                          "AeFlickerMode": controls.AeFlickerModeEnum.Manual,
                          "AeFlickerPeriod": 8333})

def capture():
    req = cam.capture_request()
    frame, md = req.make_array("main"), req.get_metadata()
    req.release()
    return frame, md["ExposureTime"] / 1000.0, md["AnalogueGain"], md

# ---- ②～⑤ CPU 전처리 (학습과 같은 코드) ----
def preprocess(frame, led_on):
    roi = frame[ROI_TOP:]                                  # 크롭 (복사 없음)
    if led_on:
        roi = cv2.multiply(roi, FF_SRGB, dtype=cv2.CV_8U)  # 플랫필드: 곱셈 한 번
    l, a, b = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB))
    roi = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    return [roi[:, x:x + 672] for x in (0, 608)], (0, 608)  # 타일 2장, 겹침 64px

# ---- ⑥ Hailo 추론 ----
def infer(tiles):
    out = []
    for t in tiles:
        res = hailo.run(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))  # uint8 RGB 그대로
        dets = [[x1 * 672, y1 * 480, x2 * 672, y2 * 480, s, c]
                for c, boxes in enumerate(res) for (y1, x1, y2, x2, s) in boxes]
        out.append(np.array(dets, np.float32).reshape(-1, 6))
    return out

# ---- 메인 루프 ----
def on_frame(confirmer, led, pose):
    frame, exp_ms, gain, md = capture()
    cmd = led.step(exp_ms, gain, md["SensorTimestamp"] / 1e9, roi_mean(frame))  # 노출×게인 판단
    apply_led_and_camera(cmd)                              # PWM + set_led_mode, 전환 직후 프레임 버림
    if not cmd.use_frame or not quality_ok(frame, cmd.frame_led_on):
        return
    tiles, xs = preprocess(frame, cmd.frame_led_on)
    dets = merge_tiles(infer(tiles), xs)                   # IoMin 병합
    dets = filter_detections(dets, SPEC_MASK)              # 방해 클래스 제거, 반사 마스크 +0.2
    for d in dets:
        ground = undistort_and_project(bottom_center(d) + (0, ROI_TOP), pose)   # 바닥 → 지도
        if confirmer.add(d.cls, ground, need=3, window=5, radius=0.05):
            publish_found(d.cls, ground, frame)            # 정지, 알림, 대시보드 표시
```

전체 구현은 `code/lost_item_preprocessing_claude/`에 있습니다(`run_demo.py --live`가 위 흐름을 그대로 실행).

---

**한 줄 요약:** 라즈베리 파이 5에서는 **① 원본 해상도 유지(크롭 + 타일링), ② 촬영 단계에서 LED·노출·초점 제어와 ISP 노이즈 제거(샤프닝은 끔), ③ 고정된 반사 위치를 이용한 마스크 + 다중 프레임 확정**이 성능을 좌우합니다. CPU에는 가벼운 연산(플랫필드 곱셈, CLAHE)만 남기고 YOLO는 Hailo NPU에 맡깁니다.
