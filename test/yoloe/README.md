# YOLOE 실습 — 무엇이든 찾는 실시간 탐지

YOLOE(Real-Time Seeing Anything)는 **학습 때 보지 못한 물체도 찾을 수 있는** 오픈 어휘 탐지·분할 모델입니다.
기존 YOLO는 학습한 80개 클래스만 찾지만, YOLOE는 **무엇을 찾을지 실행할 때 알려줄 수 있어요.**

## 세 가지 모드 한눈에 보기

| 모드 | 무엇을 주나 | 핵심 기법 | 이럴 때 쓴다 |
|---|---|---|---|
| `text` | 물체 이름 (`"cup"`, `"bus"`) | RepRTA | 찾을 물체의 이름을 알 때 |
| `visual` | 예시 박스 | SAVPE | 말로 설명하기 어려운 물체 (특정 부품 등) |
| `free` | 아무것도 안 줌 | LRPC | 이미지에 뭐가 있는지 전부 보고 싶을 때 |

```
text   : "person", "bus"  → [텍스트 인코더] → 임베딩 → 분류 헤드에 주입 → 그 물체만 탐지
visual : 예시 박스         → [시각 프롬프트 인코더] → 임베딩 → 비슷한 물체 탐지
free   : (없음)            → 내장 어휘 약 4,585개 중 각 영역에 맞는 이름을 스스로 선택
```

## 폴더 구성

| 파일 | 설명 |
|---|---|
| `yoloe_practice.py` | 실습 코드 (주석에 원리 설명) |
| `yoloe-11s-seg.pt` | 텍스트·시각 프롬프트용 모델 (첫 실행 시 자동 다운로드) |
| `yoloe-11s-seg-pf.pt` | 프롬프트 프리 전용 모델 (`pf` = prompt-free) |
| `mobileclip_blt.ts` | 텍스트 인코더, 약 600MB (텍스트 모드 첫 실행 시 다운로드) |
| `bus.jpg`, `zidane.jpg` | 예제 이미지 |
| `runs/text`, `runs/visual`, `runs/free` | 결과 이미지 |

## 실행

`test` 폴더에서:

```powershell
py yoloe\yoloe_practice.py --mode text
py yoloe\yoloe_practice.py --mode visual
py yoloe\yoloe_practice.py --mode free
py yoloe\yoloe_practice.py --mode all       # 세 모드 모두
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--mode` | `all` | `text` / `visual` / `free` / `all` |
| `--source` | 예제 이미지 | 이미지 경로, URL, 웹캠 번호(`0`). 여러 개를 띄어쓰기로 나열할 수 있어요 |
| `--classes` | `person bus glasses tie` | 텍스트 모드에서 찾을 물체 이름 |
| `--refer` | 없음 | 시각 모드에서 예시 박스를 지정할 참조 이미지 |
| `--conf` | `0.25` | 신뢰도 기준. 낮추면 더 많이 찾고, 높이면 확실한 것만 남아요 |

### 예시

```powershell
# 찾을 물체 지정 (두 단어 이상은 따옴표로 묶기)
py yoloe\yoloe_practice.py --mode text --classes tie "traffic light" "stop sign"

# 내 사진 여러 장
py yoloe\yoloe_practice.py --mode text --source "C:\Users\작업용\Pictures\a.jpg" "C:\Users\작업용\Pictures\b.jpg" --classes cup laptop

# 확실한 것만 보기
py yoloe\yoloe_practice.py --mode text --classes person --conf 0.5

# 웹캠 실시간 (종료: 터미널에서 Ctrl+C)
py yoloe\yoloe_practice.py --mode text --source 0 --classes person "cell phone"
```

## 모드별 상세

### 1. 텍스트 프롬프트 (`text`)

```python
model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(names, model.get_text_pe(names))  # 텍스트 → 임베딩 → 분류 헤드 재구성
results = model.predict(source)
```

- `set_classes`로 클래스를 한 번 정해두면, 이후 추론 속도는 일반 YOLO와 거의 같아요.
- 결과의 클래스 번호는 `--classes`에 적은 순서와 같아요.

**프롬프트 작성 요령 (직접 시험해 본 결과)**

| 프롬프트 | 결과 |
|---|---|
| `glasses` | ✅ 찾음 |
| `find the glasses` | ❌ 문장이라 실패 |
| `man wearing glasses` | ❌ 관계 설명은 약함 |
| `안경` | ❌ 한국어는 인식하지 못함 |
| `soccer player` | ❌ 역할 대신 `person`처럼 눈에 보이는 물체 이름을 써야 함 |

→ **영어 명사** 또는 짧은 **형용사+명사**(`red cup`)로 쓰세요.

### 2. 시각 프롬프트 (`visual`)

```python
visual_prompts = dict(
    bboxes=np.array([[221.52, 405.8, 344.98, 857.54],   # 박스 0: 사람
                     [120, 425, 160, 445]]),            # 박스 1: 안경
    cls=np.array([0, 1]),                               # 각 박스의 클래스 번호
)
model.predict(source, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor)
```

- 박스 좌표 `[x1, y1, x2, y2]`는 `bus.jpg` 기준이에요. 다른 이미지에서는 코드의 좌표를 바꿔야 해요.
  - 좌표는 그림판에서 마우스를 올리면 왼쪽 아래에 표시돼요.
- 결과 이름은 `object0`, `object1`처럼 나와요. 시각 프롬프트에는 이름이 없기 때문이에요.
- `--refer 참조.jpg`를 주면 **참조 이미지**에 지정한 박스의 물체를 **다른 이미지**(`--source`)에서 찾아요.
- 박스 좌표가 한 장 기준이라 이 모드는 `--source`의 첫 번째 이미지만 사용해요.

### 3. 프롬프트 프리 (`free`)

```python
model = YOLOE("yoloe-11s-seg-pf.pt")   # 내장 어휘가 들어있는 전용 모델
results = model.predict(source)
```

- `bus.jpg`에서는 person, bus 말고도 trench coat, leather shoe, stop sign 등을 찾아요.
- 넓게 찾는 대신 이름이 가끔 부정확할 수 있어요.

## 결과 읽기

터미널 출력 예:

```
  - person          conf=0.92  box=(672, 389, 810, 879)
  - bus             conf=0.85  box=(23, 232, 802, 740)
```

| 값 | 의미 |
|---|---|
| `conf` | 신뢰도 (0~1) |
| `box` | `(x1, y1, x2, y2)` 왼쪽 위·오른쪽 아래 픽셀 좌표 |
| 마스크 | 결과 이미지에 색으로 칠해진 영역 (`-seg` 모델이라 분할 마스크까지 나와요) |

코드에서는 `r.boxes.xyxy`, `r.boxes.conf`, `r.boxes.cls`, `r.masks.data`, `r.names`로 꺼내요.

## 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| 아무것도 못 찾음 | 영어 명사로 바꾸거나 `--conf 0.1`로 낮춰보기 |
| `파일을 찾을 수 없습니다` | 경로는 명령을 실행한 위치 기준이에요. 전체 경로를 쓰세요 |
| `torch.jit.load ... Python 3.14` 경고 | 텍스트 인코더를 불러올 때 나오는 경고예요. 결과에는 영향이 없어요 |
| 첫 실행이 오래 걸림 | 텍스트 인코더(약 600MB)를 내려받는 중이에요. 처음 한 번만 받아요 |

## 실습 과제

1. `"red car"`, `"person wearing hat"`처럼 수식어를 붙이면 결과가 어떻게 달라질까?
2. 시각 모드에서 박스 좌표를 바꿔 다른 물체를 예시로 줘보자.
3. 같은 이미지를 세 모드로 돌려 결과를 비교해보자.
4. 코드의 `MODEL_SIZE`를 `s` → `l`로 바꿔 속도와 정확도 차이를 확인해보자.
