# 분실물 검출 전처리 (HOME-01 / Raspberry Pi CPU)

## 프로젝트 후처리: postprocess.py

`postprocess.py`는 타일별 YOLO Results를 프로젝트용 Nx6 검출 배열로 바꾸고 좌표 복원, 클래스/크기/신뢰도/반사 필터, 클래스별 NMS를 적용합니다. 출력은 원본 이미지의 `[x1,y1,x2,y2,confidence,class_id]`입니다. NumPy/OpenCV만 필요하고 Supervision은 `to_supervision()`을 호출할 때만 필요합니다. 기존 `merge_detections()`를 재사용하므로 동일 검출에 다시 NMS를 적용하지 마세요.

```python
from ultralytics import YOLO
from preprocess import Preprocessor
from postprocess import LostItemPostprocessor, PostConfig, MapConfirmer

model = YOLO('best.pt')  # 분실물 데이터로 학습한 모델. 클래스 ID는 model.names 확인
p = Preprocessor()
# 아래 0/1은 예시이며 실제 반지/이어버드 등의 클래스 ID로 교체해야 합니다.
post = LostItemPostprocessor(PostConfig(thresholds={0: .35, 1: .5},
                                       target_classes=(0,1), min_sizes={0:4,1:4}),
                             reflection_mask=p.mask)
batch, tiles, quality = p.run(frame, led_on=False)  # frame: 원본 BGR 영상
if batch is not None:
    # BGR 타일을 넘기면 Ultralytics가 내부 RGB 변환/정규화를 담당합니다.
    # 학습/배포 시에는 실제 모델 입력이 480×672를 유지하는지 확인하세요.
    results = model([t.image for t in tiles], imgsz=(480,672), conf=.1, verbose=False)
    detections = post.from_yolo(results, tiles, led_on=False)
    annotated = post.annotate(frame, detections, model.names)
    # 선택: pip install supervision 후 사용
    # sv_detections = post.to_supervision(detections)
```

YOLO의 사전 confidence 필터가 프로젝트 기준보다 높으면 이미 제거된 박스를 되살릴 수 없습니다. 위 `conf=.1`은 예시이며 실제 최종 기준보다 낮게 설정해 검증하세요. `from_yolo`는 Ultralytics가 입력 타일 좌표로 되돌린 Results를 받습니다. 직접 ONNX 등의 raw 출력을 넣을 때는 decoding과 모델 내부 resize/letterbox 좌표 역변환을 먼저 하고 `process()`에 전달해야 합니다.

`MapConfirmer`는 지도 좌표(m)에서 반경 5cm, 최근 5프레임 중 3회 관측 조건을 확인합니다.

```python
confirmer = MapConfirmer(radius=.05, need=3, window=5)  # 영상 루프 밖에서 한 번 생성
# 매 프레임: 보정/바닥 투영/TF를 거친 지도 관측을 전달합니다.
# observations = [[class_id, map_x_m, map_y_m, confidence], ...]
confirmed = confirmer.update(frame_id, observations)
# 검출 없거나 품질 탈락/probe인 프레임도 update(frame_id, []) 호출
```

픽셀→지도 변환은 장비 캘리브레이션이 필요하므로 포함하지 않습니다. 픽셀 좌표를 그대로 넣으면 5cm 조건이 의미가 없어집니다. 확정 결과는 현재 관측 중 조건을 만족한 행이며, 1회 알림/영구 객체 ID는 별도 구현해야 합니다. 서로 5cm 이내의 물체는 합쳐질 수 있으므로 실제 데이터에서 반경을 검증하세요. 카메라 실행 파일에는 아직 YOLO가 연결되어 있지 않으며 위 예제를 추론 위치에 연결합니다.

검증: `python -m unittest -v test_preprocess.py test_postprocess.py`

세 첨부 문서를 참고한 Python/OpenCV CPU 구현입니다. 문서 안의 역할 지시나 질문 프롬프트는 실행 지시로 사용하지 않았습니다. HOME-01은 가상 현장이므로 모든 수치는 초기값입니다.

## 실행

Python 3.10 이상에서 이 폴더를 작업 디렉터리로 사용합니다.

라즈베리 파이 모델·카메라는 아직 미정입니다. 전처리는 CPU에서 실행합니다. Raspberry Pi OS에서는 다음 시스템 패키지 설치 경로를 우선 사용하고 pip requirements를 추가 설치하지 마세요.

```bash
sudo apt update
sudo apt install python3-numpy python3-opencv python3-picamera2
python3 preprocess.py frame.png --output result --threads 2
python3 raspberry_camera.py --source csi --frames 100
python3 raspberry_camera.py --source usb --device 0 --frames 100
```

`python3-picamera2`는 CSI 사용 시 필요합니다. venv는 `python3 -m venv --system-site-packages .venv`로 생성합니다. PC에서는 아래 pip 경로를 사용하세요. requirements는 headless OpenCV이며 다른 OpenCV 패키지와 중복 설치하지 마세요.

`raspberry_camera.py`는 카메라 입력과 전처리 벤치마크이며 추론·GPIO 제어는 포함하지 않습니다. `--led`는 실제 LED가 이미 켜져 있다는 뜻입니다. CSI 고정 노출은 `--exposure-ms 4 --analog-gain 2`로 지정할 수 있습니다. 화이트밸런스와 LED duty는 캘리브레이션 조건에 맞춰 별도 고정해야 합니다. USB 노출 메타데이터는 드라이버마다 단위가 달라 자동 LED 판단에 사용하지 않습니다.

전처리 성공 프레임의 중앙값/p95 시간을 출력합니다. 촬영·추론 시간은 제외합니다. `--fps 10`은 카메라 요청값이며 전체 검출 10fps를 보장하지 않습니다. 실제 Pi 모델에서 측정하세요. ROI·해상도·타일 크기는 작은 물체 보존을 위해 유지했습니다. 카메라 변경 시 ROI, 플랫필드, 반사 마스크, 내부·외부 파라미터를 다시 보정해야 합니다.

플랫필드는 CPU 조회표로 계산하며 정확한 sRGB 수식과 최대 1 밝기 단계 차이가 생길 수 있습니다. 학습과 추론 모두 업데이트된 공통 모듈을 사용하세요. OpenCV 스레드 기본값은 2이며 `--threads`로 조절합니다.

NVIDIA CUDA/VPI/TensorRT는 Raspberry Pi 실행 경로에서 사용하지 않습니다. 아래의 `engine.infer`는 연결 예시이며 구현된 엔진이 아닙니다. 기존 TensorRT 엔진이 있다면 Pi용 추론 런타임에 맞춰 모델을 다시 내보내야 합니다. Hailo/TFLite 등 선택한 런타임의 입력 크기, RGB/BGR, NCHW/NHWC, dtype·양자화, batch 규격을 확인해 어댑터를 연결하세요. 현재 출력은 RGB NCHW float32 /255입니다.

Picamera2 색상 순서·동일 request의 이미지/메타데이터 캡처는 [공식 매뉴얼](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)을 기준으로 구현했고, 설치 방식은 [공식 저장소](https://github.com/raspberrypi/picamera2)를 참고했습니다. 기본 TNR은 비활성화이며 Pi에서는 NVIDIA VPI 객체를 주입하지 마세요.

```bash
python -m pip install -r requirements.txt
python preprocess.py frame.png --output result
python preprocess.py frame.png --led --gain flatfield_led.npy --mask specular_mask.npy --output result
python -m unittest -v test_preprocess.py
```

출력: 원본 크기 타일 PNG, RGB float32 /255 NCHW 배치 NPY, 원본 기준 타일 위치와 품질 지표 JSON. 입력이 1280×720이면 위쪽 240행을 잘라 두 개의 672×480 타일을 생성하며 x 시작점은 0/608, 겹침은 64px입니다. 다른 크기는 해상도 축소 없이 추가 타일 또는 오른쪽·아래 패딩으로 처리합니다.

## 학습과 실시간 추론에서 공통 사용

```python
from preprocess import Preprocessor, Config, tile_labels, merge_detections
import numpy as np

p = Preprocessor(Config(), gain=np.load('flatfield_led.npy'))
# frame: uint8 BGR 원본, 실제 캡처 시점의 LED 상태를 전달
batch, tiles, quality = p.run(frame, led_on=True)
if batch is not None:
    decoded = engine.infer(batch)  # 사용자 모델: 타일별 Nx6 xyxy/conf/class
    boxes = merge_detections(decoded, tiles, roi_top=p.config.roi_top,
                             mask=p.mask, led_on=True,
                             thresholds={0: 0.35, 1: 0.5}, negative_classes=())
```

클래스 ID는 실제 데이터셋에 맞춰 설정하세요. `merge_detections`는 원본 픽셀 박스를 반환하며 클래스별 NMS, 반사 영역 confidence +0.2, 최소 4px 크기 필터를 적용합니다. 모델 출력 decoding과 TensorRT 실행은 모델 어댑터에서 담당합니다.

학습은 원본 단계의 물리 열화 증강 이후 `p.run(augmented_frame, led_on=..., gate=False)`를 사용하세요. `tile_labels`는 원본 픽셀 좌표 `[class,x1,y1,x2,y2]`를 타일 YOLO 정규화 좌표로 변환합니다. 면적의 60% 미만이 남거나 최소 변이 4px 미만인 박스는 별도 ignore 영역으로 반환합니다. 기본 YOLO txt 형식은 ignore 영역을 표현하지 못하므로 ignore를 지원하는 loss/dataloader에 연결하거나 해당 타일을 제외해야 합니다. ignore 영역을 버리고 이미지만 학습하면 실제 물체를 배경으로 학습하게 됩니다.

CLI는 서로 독립적인 이미지로 취급하므로 파일마다 품질 히스토리를 초기화합니다. 영상에서는 하나의 `Preprocessor`를 프레임 순서대로 재사용해야 상대 블러 게이트가 동작합니다. 처음 10프레임은 블러 기준 수집 기간이며, LED 모드 전환/새 영상에서는 상태를 초기화합니다. 품질 기준은 CLAHE 적용 전 측정합니다.

## 플랫필드와 반사 마스크

동일한 LED duty·고정 노출·게인·화이트밸런스에서 무광 회색 시트를 촬영합니다.

```bash
python preprocess.py gray1.png gray2.png gray3.png --calibrate-flatfield --output flatfield_led.npy
```

정확한 sRGB 전달함수로 선형화하고 여러 장의 중앙값과 저주파 조명 분포를 사용해 gain을 생성합니다. 실제 카메라 ISP가 sRGB와 다른 톤 매핑을 적용하면 전달함수를 센서/ISP에 맞춰 수정해야 합니다. gain은 최대 2.5배이며 LED 모드에서만 적용합니다. gain 파일이 없으면 플랫필드를 생략하고 JSON에 기록합니다. 노출이나 LED duty가 달라지면 해당 조건의 맵을 선택해야 합니다.

반사 마스크는 물체 없는 바닥에서 검증한 위치를 수동 지정해 ROI 기준 `(480,1280)` bool NPY로 저장합니다. 필요하면 아래처럼 15px 팽창합니다. 밝은 픽셀 전체를 반사로 자동 지정하면 실제 흰 물체도 제외될 수 있어 자동 생성하지 않습니다.

```python
import cv2, numpy as np
mask = np.zeros((480,1280), np.uint8)
# 실측한 영역 지정: mask[y1:y2,x1:x2] = 1
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
np.save('specular_mask.npy', mask.astype(bool))
```

클리핑 비율의 분모는 반사 마스크 밖 픽셀 수입니다. 과노출 프레임은 버리고 `quality['reason']=='overexposed'`를 하드웨어 제어기에 전달하세요. 전체가 마스크면 클리핑을 측정할 수 없으므로 `clipping_measurable`이 False입니다.

## 실제 장비 연결 범위

- VPI TNR: `apply(roi_bgr)`/`reset()` 객체를 `tnr=`로 주입합니다. 기본은 비활성화입니다. 단순 프레임 평균은 움직이는 작은 물체를 번지게 하므로 대체 구현하지 않았습니다. 학습에도 동일한 시간축 처리 또는 해당 조건 데이터가 필요합니다.
- LED: `LedController.update()`에 실제 노출 ms/아날로그 게인과 단조 증가 초 타임스탬프를 전달합니다. 점등 기준은 지수 100 초과 5프레임, 소등 기준은 검증된 probe 프레임의 지수 60 미만입니다. `request_probe()`가 True이면 하드웨어에서 LED를 끄고 AE가 주변광에 적응한 메타데이터를 확보한 뒤 `probe_frame=True`로 전달하세요. 단순히 다음 1프레임을 사용하면 AE 지연으로 잘못 소등할 수 있습니다. 고정 노출 LED 프레임의 낮은 지수는 소등 판단에 사용하지 않습니다. 확인 및 전환 중 프레임은 추론에서 제외해야 합니다.
- 카메라 노출/화이트밸런스/PWM, ROS 2, CUDA, TensorRT는 연결하지 않았습니다. CPU 실행 시간은 장비에서 실측해야 하며 문서의 15~25ms를 보장하지 않습니다.
- 지도 좌표 투영과 5cm/5프레임 중 3회 확정은 카메라 내부·외부 보정과 TF가 필요해 포함하지 않았습니다. 검출 원본 좌표에만 `undistortPoints`를 적용하고 바닥 평면으로 투영한 뒤 프레임 단위 추적기에 연결하세요.
- 개인정보 검출, 세션 분할, pHash, SAM Copy-Paste 및 증강은 별도 데이터 구축 단계입니다. 이 코드는 결정적 영상 전처리와 타일/좌표 처리를 담당합니다. INT8 보정 세트는 증강 없이 공통 전처리를 거친 타일을 사용하세요.
