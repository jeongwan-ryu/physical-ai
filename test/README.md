# 비전 AI 실습

| 폴더 | 모델 | 배우는 것 | 자세한 설명 |
|---|---|---|---|
| `yoloe/` | YOLOE | 텍스트 / 이미지 / 프롬프트 없이 물체 찾기 | [yoloe/README.md](yoloe/README.md) |
| `sam2/` | SAM 2 | 점·박스로 가리킨 물체 분할 | [sam2/README.md](sam2/README.md) |
| `siglip2/` | SigLIP 2 | 텍스트·이미지 임베딩과 코사인 유사도 | [siglip2/README.md](siglip2/README.md) |
| `ocr/` | EasyOCR | 이미지 속 글자 읽기 (한국어+영어) | [ocr/README.md](ocr/README.md) |
| `mediapipe/` | MediaPipe | 손가락으로 숫자 세기 (웹캠) | [mediapipe/README.md](mediapipe/README.md) |

## 1. 설치 (처음 한 번)

```powershell
py -m pip install -U ultralytics transformers sentencepiece protobuf easyocr mediapipe

# EasyOCR이 창 띄우기가 안 되는 OpenCV를 같이 설치하므로 원래대로 복구 (SAM2 click 모드용)
py -m pip uninstall -y opencv-python-headless
py -m pip install --force-reinstall --no-deps opencv-python
```

> Windows에서는 `python` 대신 `py`로 실행하세요.
> 모델과 예제 이미지는 처음 실행할 때 자동으로 내려받아요.
> 이미지 경로(`--source`, `--images`)는 **명령을 실행한 위치 기준**이에요. 헷갈리면 전체 경로를 쓰세요.
> 아래 예시의 `C:\Users\...\desk.jpg` 같은 경로는 **내 사진 경로로 바꿔서** 쓰세요.

## 2. 실행

먼저 이 폴더로 이동하세요.

```powershell
cd test
```

### YOLOE — 물체 찾기

```powershell
py yoloe\yoloe_practice.py --mode text      # 텍스트로 지정한 물체 찾기
py yoloe\yoloe_practice.py --mode visual    # 예시 박스와 비슷한 물체 찾기
py yoloe\yoloe_practice.py --mode free      # 지정 없이 보이는 것 전부
```

찾을 물체 바꾸기:

```powershell
py yoloe\yoloe_practice.py --mode text --classes cup laptop "cell phone"
py yoloe\yoloe_practice.py --mode text --source "C:\Users\작업용\Pictures\desk.jpg" --classes cup
```

- 물체 이름은 **영어 명사**로 넣어야 해요. 문장이나 한국어는 인식하지 못해요.
- 결과는 `yoloe\runs\`에 저장돼요.

### SAM 2 — 물체 분할

```powershell
py sam2\sam2_practice.py --mode click   # ★ 마우스로 직접 찍어보기
py sam2\sam2_practice.py --mode all     # 점 / 박스 / 여러 물체 / YOLOE 조합 / 전체 분할
```

click 모드 조작법:

| 입력 | 동작 |
|---|---|
| 좌클릭 | 물체로 지정 |
| 우클릭 | 배경으로 지정 (마스크 깎기) |
| 드래그 | 박스로 지정 |
| `r` / `s` / `q` | 초기화 / 저장 / 종료 |

- 결과는 `sam2\runs\`에 저장돼요.

### SigLIP 2 — 임베딩 & 유사도

```powershell
py siglip2\siglip2_practice.py
py siglip2\siglip2_practice.py --texts "a photo of a red bus" "a photo of a cat"
py siglip2\siglip2_practice.py --images yoloe\zidane.jpg sam2\bus.jpg
```

- 유사도 표는 터미널에 출력돼요.
- 히트맵은 `siglip2\runs\`에 저장돼요.

### OCR — 글자 읽기

```powershell
py ocr\ocr_practice.py --mode all                      # 읽기 / 2단계 / 강건성 / 단어 찾기
py ocr\ocr_practice.py --mode find --find 송장번호     # 특정 단어 찾고 값 뽑기
py ocr\ocr_practice.py --mode basic --source "C:\Users\작업용\Pictures\receipt.jpg"
```

- 결과는 `ocr\runs\`에 저장돼요.

### MediaPipe — 손가락 숫자 세기

```powershell
py mediapipe\finger_count.py            # 웹캠에 손가락을 펴서 보여주기 (q: 종료)
py mediapipe\finger_count.py --debug    # 관절 번호와 판단 값까지 표시
py mediapipe\finger_count.py --demo     # 웹캠 없이 예제 사진으로 테스트
```

- 결과는 `mediapipe\runs\`에 저장돼요.

## 3. 더 공부하기

각 `.py` 파일 맨 위에 모델의 원리가, 맨 아래에 **실습 과제**가 주석으로 정리되어 있어요.
코드를 직접 고쳐가며 결과가 어떻게 달라지는지 확인해 보세요.
