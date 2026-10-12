# SigLIP 2 실습 — 텍스트·이미지 임베딩과 코사인 유사도

SigLIP 2는 Google이 만든 **비전-언어 모델**입니다.
이미지와 텍스트를 **같은 벡터 공간**에 올려놓아서 "이 사진과 이 문장이 얼마나 비슷한가?"를 숫자로 계산할 수 있어요.

```
"a photo of a bus" ──▶ [Text Encoder]  ──▶ 768차원 벡터 ─┐
                                                          ├─▶ 코사인 유사도
   버스 사진        ──▶ [Image Encoder] ──▶ 768차원 벡터 ─┘
```

YOLOE의 텍스트 프롬프트도 같은 원리(텍스트 임베딩)로 동작해요.

## 핵심 개념

### 임베딩 (Embedding)
텍스트나 이미지를 **고정 길이 숫자 벡터**로 바꾼 것입니다. 숫자 하나하나의 의미는 사람이 해석할 수 없지만, 벡터 사이의 **각도**가 의미의 유사성을 나타내도록 학습되어 있어요.

### 코사인 유사도 (Cosine Similarity)

```
cos(θ) = (a · b) / (|a| |b|)
```

| 값 | 의미 |
|---|---|
| `1` | 같은 방향 (의미가 같음) |
| `0` | 직각 (관계없음) |
| `-1` | 반대 방향 |

벡터를 길이 1로 **정규화**하면 내적이 곧 코사인 유사도예요. 코드에서는 이렇게 계산해요.

```python
F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
```

### CLIP과 SigLIP의 차이

| | CLIP | SigLIP |
|---|---|---|
| 학습 방식 | softmax | sigmoid |
| 질문 | "N개 문장 중 정답은?" | "이 (이미지, 문장) 쌍이 맞나? 예/아니오" |
| 한 이미지의 확률 합 | 항상 1 | 1이 아니어도 됨 (여러 개가 맞거나 전부 틀릴 수 있음) |

SigLIP 2는 여기에 **다국어 학습**과 위치 파악 능력을 더한 개선판이에요. 그래서 한국어도 어느 정도 이해해요.

## 폴더 구성

| 파일 | 설명 |
|---|---|
| `siglip2_practice.py` | 실습 코드 (주석에 원리 설명) |
| `bus.jpg`, `zidane.jpg`, `cats.jpg` | 예제 이미지 (첫 실행 시 자동 다운로드) |
| `runs/image_text.png` | 이미지↔텍스트 유사도 히트맵 |
| `runs/text_text.png` | 텍스트↔텍스트 유사도 히트맵 |
| `runs/image_image.png` | 이미지↔이미지 유사도 히트맵 |

모델(`google/siglip2-base-patch16-224`, 약 1.5GB)은 이 폴더가 아니라 허깅페이스 공용 캐시(`C:\Users\작업용\.cache\huggingface`)에 저장돼요.

## 실행

처음 한 번 설치:

```powershell
py -m pip install -U transformers sentencepiece protobuf
```

`test` 폴더에서:

```powershell
py siglip2\siglip2_practice.py
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--texts` | 문장 4개 (bus / two men wearing ties / two cats / dog) | 실험 2에서 이미지와 비교할 문장 |
| `--images` | 예제 이미지 3장 | 비교할 이미지 경로 |

### 예시

```powershell
# 내 문장으로 비교
py siglip2\siglip2_practice.py --texts "a photo of a red bus" "a photo of a blue bus"

# 다른 폴더의 예제 이미지로 비교 (그대로 실행 가능)
py siglip2\siglip2_practice.py --images yoloe\zidane.jpg sam2\bus.jpg

# 내 사진으로 비교
py siglip2\siglip2_practice.py --images "C:\Users\작업용\Pictures\a.jpg" "C:\Users\작업용\Pictures\b.jpg"
```

## 실험 구성과 실제 결과

### 실험 1. 임베딩 들여다보기
- 텍스트와 이미지 모두 **768차원** 벡터예요. 차원이 같으니 바로 비교할 수 있어요.
- 정규화하기 전 벡터 길이는 텍스트가 약 18~24, 이미지가 약 11로 서로 달라요. 코사인 유사도는 방향만 보므로 상관없어요.

### 실험 2. 이미지 ↔ 텍스트 (zero-shot 분류)

| | a bus | two men wearing ties | two cats | a dog |
|---|---|---|---|---|
| bus.jpg | **+0.121** | -0.004 | -0.011 | +0.001 |
| zidane.jpg | +0.006 | **+0.088** | +0.003 | +0.004 |
| cats.jpg | -0.022 | -0.013 | **+0.129** | +0.026 |

- 이미지↔텍스트 코사인 값은 원래 작아요(-0.1~0.2). **절대값보다 순위**를 보세요.
- 코사인 값을 확률로 바꾸는 식은 `sigmoid(cos × exp(logit_scale) + logit_bias)`예요.
  - `logit_scale`과 `logit_bias`는 학습된 값으로 모델 안에 들어 있어요.
- 맞는 문장의 sigmoid 확률도 4%, 9%처럼 낮게 나와요. SigLIP이 엄격하게 판단하기 때문이에요.
- 같은 값을 softmax(CLIP 방식)로 바꾸면 각 행의 합이 1(100%)이 되는 것도 비교해서 보여줘요.

### 실험 3. 텍스트 ↔ 텍스트 (동의어, 다국어)
- `a cat` 기준: kitten **0.97** > dog **0.94** > 고양이 사진 **0.93** > bus **0.87**
- 텍스트끼리는 전부 0.8 이상으로 높게 나와요. 여기서도 **순서**를 보세요.
- 한국어 문장으로 이미지를 찾아도 정답을 맞혀요.
  - `버스` → bus.jpg
  - `넥타이를 맨 남자` → zidane.jpg
  - `고양이 두 마리` → cats.jpg

### 실험 4. 이미지 ↔ 이미지
- 원본과 **좌우 반전한 이미지**의 유사도는 **0.958**이에요. 픽셀은 달라도 의미가 같기 때문이에요.
- 서로 다른 사진끼리는 0.4~0.5 정도예요.
- 이것이 **유사 이미지 검색**의 원리예요.

## 코드 핵심

```python
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

# 텍스트 임베딩 — SigLIP은 64토큰 padding으로 학습했으므로 꼭 맞춰야 해요
t = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt")
text_emb = model.get_text_features(**t)

# 이미지 임베딩
i = processor(images=images, return_tensors="pt")
image_emb = model.get_image_features(**i)

# 코사인 유사도 행렬
sim = F.normalize(image_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
```

## 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| `No module named 'transformers'` | `py -m pip install -U transformers sentencepiece protobuf` |
| `파일을 찾을 수 없습니다` | 경로는 명령을 실행한 위치 기준이에요. 전체 경로를 쓰세요 |
| 확률이 너무 낮게 나옴 | 정상이에요. 순위로 판단하거나 문장을 더 구체적으로 써보세요 |
| 첫 실행이 오래 걸림 | 모델(약 1.5GB)을 내려받는 중이에요. 처음 한 번만 받아요 |

## 실습 과제

1. `"a photo of a bus"`와 `"bus"`를 함께 넣어보자. 문장 형태가 점수를 바꿀까?
2. `"a photo of a red bus"`와 `"a photo of a blue bus"` — 색깔을 구분할까?
3. `"a photo of a man to the left of a bus"`처럼 **위치 관계**는 이해할까? (CLIP 계열이 약한 부분)
4. 내 사진 여러 장을 `--images`로 넣고 가장 비슷한 사진 쌍을 찾아보자.
