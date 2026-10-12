"""
SigLIP 2 실습 코드 — 텍스트/이미지 임베딩과 코사인 유사도
==========================================================

SigLIP 2 는 Google 이 만든 '비전-언어' 모델이다.
이미지와 텍스트를 '같은 벡터 공간'에 올려놓는 것이 핵심이다.

    "a photo of a bus"  ──▶ [Text Encoder]  ──▶ 768차원 벡터 ─┐
                                                              ├─ 코사인 유사도 → 얼마나 비슷한가?
    🖼 버스 사진         ──▶ [Image Encoder] ──▶ 768차원 벡터 ─┘

  - 의미가 비슷하면 벡터 방향이 비슷하다 → 코사인 유사도가 높다.
  - 이 덕분에 학습 없이도 "이 사진은 어떤 문장과 가장 비슷한가?"로 분류할 수 있다 (zero-shot).
  - 앞에서 실습한 YOLOE 의 텍스트 프롬프트도 같은 원리(MobileCLIP 텍스트 임베딩)를 쓴다.

임베딩(embedding) 이란?
  텍스트나 이미지를 고정 길이 숫자 벡터로 바꾼 것. 숫자 하나하나의 의미는 사람이 해석할 수 없지만,
  벡터끼리의 '거리/각도'가 의미의 유사성을 나타내도록 학습되어 있다.

코사인 유사도(cosine similarity) 란?
  두 벡터 사이 각도의 코사인 값.  cos(θ) = (a · b) / (|a| |b|)
    1  → 완전히 같은 방향 (의미 같음)
    0  → 직각 (관계 없음)
   -1  → 반대 방향
  벡터 길이를 1로 정규화(normalize)해두면 '내적(a · b)'이 곧 코사인 유사도가 된다.
  ※ 이미지-텍스트 사이의 코사인 값은 보통 -0.1 ~ 0.2 정도로 작다. 절대값보다 '순위'를 보자.
    (텍스트-텍스트, 이미지-이미지 사이 값은 훨씬 크게 나온다. 같은 종류끼리라 그렇다.)

CLIP 과 SigLIP 의 차이 (시험에 자주 나오는 포인트):
  - CLIP   : softmax 로 학습. "N개 문장 중 정답은 하나" → 확률 합이 1
  - SigLIP : sigmoid 로 학습. "이 (이미지, 문장) 쌍이 맞나? 예/아니오" 를 쌍마다 따로 판단
             → 확률 합이 1일 필요가 없다. 여러 문장이 동시에 맞을 수도, 전부 틀릴 수도 있다.
  - SigLIP 2 는 여기에 다국어 학습, 위치 파악(localization) 능력 등을 더해 개선한 버전이다.
    그래서 한국어 문장도 어느 정도 이해한다. (아래 실험 3에서 확인)

설치:
    py -m pip install -U transformers sentencepiece protobuf

실행:
    py siglip2_practice.py
    py siglip2_practice.py --texts "a red bus" "a man with a tie" "two cats"   # 내 문장으로
    py siglip2_practice.py --images my1.jpg my2.jpg                           # 내 이미지로

처음 실행 시 모델(~1.5GB)과 예제 이미지가 자동 다운로드된다.
결과 히트맵은 이 폴더의 runs/ 에 저장된다.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor

# 실습과 무관한 경고 메시지 숨기기 (Windows 심볼릭 링크 경고, 토크나이저 설정 경고)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
transformers.logging.set_verbosity_error()

HERE = Path(__file__).resolve().parent
# 사용자가 입력한 경로는 '실행한 위치' 기준이므로, 폴더를 옮기기 전 위치를 기억해 둔다.
START_DIR = Path.cwd()


def user_path(p):
    """명령줄로 받은 상대경로를 실행 위치 기준 절대경로로 바꾼다. (URL·웹캠 번호는 그대로)"""
    if p.isdigit() or p.startswith("http"):
        return p
    path = (START_DIR / p).resolve()
    if not path.exists():
        raise SystemExit(f"파일을 찾을 수 없습니다: {path}")
    return str(path)

os.chdir(HERE)
SAVE_DIR = HERE / "runs"

# 모델 선택
#   base-patch16-224 : 이미지를 224x224 로 줄이고 16x16 패치로 나눠 처리 (가장 가벼운 축)
#   더 크거나 해상도가 높은 모델: google/siglip2-large-patch16-256, google/siglip2-so400m-patch14-384 등
MODEL_ID = "google/siglip2-base-patch16-224"

# 예제 이미지: (파일명, URL)
DEFAULT_IMAGES = [
    ("bus.jpg", "https://ultralytics.com/images/bus.jpg"),                    # 버스 + 사람들
    ("zidane.jpg", "https://ultralytics.com/images/zidane.jpg"),              # 넥타이 맨 두 남자
    ("cats.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg"),  # 소파 위 고양이 두 마리
]

# 예제 문장. SigLIP 계열은 "a photo of ~" 형태의 문장에서 성능이 좋다 (학습 데이터가 사진 캡션이라서).
DEFAULT_TEXTS = [
    "a photo of a bus",
    "a photo of two men wearing ties",
    "a photo of two cats",
    "a photo of a dog",
]


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------
def load_images(paths):
    """경로 리스트 → PIL 이미지 리스트. 기본 예제는 없으면 내려받는다."""
    images, names = [], []
    if paths:
        for p in paths:
            images.append(Image.open(user_path(p)).convert("RGB"))
            names.append(Path(p).name)
    else:
        for name, url in DEFAULT_IMAGES:
            if not Path(name).exists():
                print(f"  다운로드: {url}")
                Path(name).write_bytes(requests.get(url, timeout=30).content)
            images.append(Image.open(name).convert("RGB"))
            names.append(name)
    return images, names


def as_tensor(out):
    """transformers 버전에 따라 get_*_features 가 텐서 또는 출력 객체를 돌려주므로 통일한다."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


@torch.no_grad()  # 추론만 하므로 기울기 계산을 끈다 (메모리/속도 절약)
def embed_texts(model, processor, texts):
    """
    텍스트 → 임베딩 벡터 [N, D]

    1) 토크나이저가 문장을 토큰 ID 로 바꾼다.
       ※ SigLIP 은 학습할 때 항상 64 토큰 길이로 padding 했으므로,
         padding="max_length", max_length=64 를 꼭 맞춰줘야 결과가 정상적으로 나온다.
    2) 텍스트 인코더(Transformer)가 토큰들을 읽고 문장 전체를 대표하는 벡터 하나를 만든다.
    """
    inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt")
    return as_tensor(model.get_text_features(**inputs))


@torch.no_grad()
def embed_images(model, processor, images):
    """
    이미지 → 임베딩 벡터 [N, D]

    1) 프로세서가 이미지를 224x224 로 리사이즈하고 정규화한다.
    2) 이미지 인코더(ViT)가 16x16 패치들을 읽고 이미지 전체를 대표하는 벡터 하나를 만든다.
    """
    inputs = processor(images=images, return_tensors="pt")
    return as_tensor(model.get_image_features(**inputs))


def cosine_matrix(a, b):
    """
    코사인 유사도 행렬 [len(a), len(b)]

    직접 계산해보면:  (a / |a|) @ (b / |b|).T
      - F.normalize: 각 벡터를 길이 1로 만든다 (L2 정규화)
      - @ (행렬곱): 모든 쌍의 내적을 한 번에 계산 → 정규화했으므로 = 코사인 유사도
    """
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


def short(text):
    """표에서 열 이름이 너무 길지 않게 공통 접두어 'a photo of' 를 생략해서 보여준다."""
    return text.removeprefix("a photo of ")


def print_matrix(mat, rows, cols, title, fmt="{:+.3f}", mark_best=True):
    """행렬을 표로 출력. mark_best=True 면 각 행의 최댓값에 ◀ 표시."""
    print(f"\n  [{title}]")
    rows = [short(r) for r in rows]
    cols = [short(c) for c in cols]
    w = max(len(r) for r in rows)
    cw = max(10, max(len(c) for c in cols))  # 열 너비 = 가장 긴 열 이름
    print("  " + " " * w + " | " + " | ".join(f"{c:>{cw}}" for c in cols))
    for r, vals in zip(rows, mat.tolist()):
        best = max(vals)
        cells = [f"{fmt.format(v):>{cw - 2}}" + (" ◀" if mark_best and v == best else "  ") for v in vals]
        print(f"  {r:<{w}} | " + " | ".join(cells))


def save_heatmap(mat, rows, cols, title, filename):
    SAVE_DIR.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]  # 한글 문장 표시용 (Windows 기본 폰트)
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(1.8 * len(cols) + 3, 0.7 * len(rows) + 2))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = SAVE_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  히트맵 저장: {path}")


# ---------------------------------------------------------------------------
# 메인 실험
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP 2 임베딩 / 코사인 유사도 실습")
    parser.add_argument("--images", nargs="+", default=None, help="비교할 이미지 경로들")
    parser.add_argument("--texts", nargs="+", default=None, help="비교할 문장들")
    args = parser.parse_args()

    print(f"모델 로드: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).eval()   # eval(): 추론 모드 (dropout 등 끔)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    images, image_names = load_images(args.images)
    texts = args.texts or DEFAULT_TEXTS

    # -----------------------------------------------------------------------
    # 실험 1. 임베딩이 어떻게 생겼나
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("실험 1. 임베딩 벡터 들여다보기")
    print("=" * 70)
    text_emb = embed_texts(model, processor, texts)
    image_emb = embed_images(model, processor, images)

    # 텍스트와 이미지 임베딩의 차원(D)이 같다 → 같은 공간에 있으므로 직접 비교 가능
    print(f"  텍스트 임베딩 shape : {tuple(text_emb.shape)}   (문장 수, 차원)")
    print(f"  이미지 임베딩 shape : {tuple(image_emb.shape)}   (이미지 수, 차원)")
    print(f"  '{texts[0]}' 의 앞 8개 값: {[round(v, 3) for v in text_emb[0, :8].tolist()]}")
    # 정규화 전 길이(norm)는 제각각이다. 코사인 유사도는 '방향'만 보므로 길이는 상관없다.
    print(f"  벡터 길이(L2 norm) — 텍스트: {[round(v, 1) for v in text_emb.norm(dim=-1).tolist()]}")
    print(f"                      이미지: {[round(v, 1) for v in image_emb.norm(dim=-1).tolist()]}")

    # -----------------------------------------------------------------------
    # 실험 2. 이미지 ↔ 텍스트 유사도 (zero-shot 분류)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("실험 2. 이미지 ↔ 텍스트 코사인 유사도 (◀ = 그 이미지에서 가장 높은 문장)")
    print("=" * 70)
    sim = cosine_matrix(image_emb, text_emb)  # [이미지 수, 문장 수]
    print_matrix(sim, image_names, texts, "cosine similarity")

    # 코사인 값을 '확률'로 바꾸기 — SigLIP 이 학습한 방식 그대로
    #   logit = cos × exp(logit_scale) + logit_bias   → sigmoid → 0~1 확률
    #   logit_scale(온도)과 logit_bias 는 학습된 값으로 모델 안에 들어 있다.
    #   작은 코사인 차이를 크게 벌려서 '맞다/아니다'를 뚜렷하게 만드는 역할.
    scale = model.logit_scale.exp().item()
    bias = model.logit_bias.item()
    probs = torch.sigmoid(sim * scale + bias)
    print(f"\n  logit_scale = exp({model.logit_scale.item():.3f}) = {scale:.1f},  logit_bias = {bias:.2f}")
    print_matrix(probs, image_names, texts, "sigmoid 확률 (각 칸이 독립적인 예/아니오 판단)", fmt="{:.1%}")
    print("\n  → 한 행의 확률을 더해보자. 1이 아니어도 된다! (softmax 를 쓰는 CLIP 과의 차이)")

    # 비교용: 같은 값에 softmax 를 쓰면 (CLIP 방식) 각 행의 합이 1이 된다
    softmax_probs = (sim * scale).softmax(dim=-1)
    print_matrix(softmax_probs, image_names, texts, "참고: softmax 로 바꿨을 때 (CLIP 방식, 행 합 = 1)", fmt="{:.1%}")
    save_heatmap(sim.numpy(), image_names, texts, "image-text cosine similarity", "image_text.png")

    # -----------------------------------------------------------------------
    # 실험 3. 텍스트 ↔ 텍스트 유사도 (동의어, 무관한 문장, 다국어)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("실험 3. 텍스트 ↔ 텍스트 코사인 유사도")
    print("=" * 70)
    t_texts = [
        "a photo of a cat",
        "a photo of a kitten",      # 동의어에 가까움 → 높을 것
        "고양이 사진",                # 한국어 (같은 의미) → SigLIP 2 는 다국어라 꽤 높게 나온다
        "a photo of a dog",         # 같은 '동물' 범주 → 중간
        "a photo of a bus",         # 무관 → 낮을 것
    ]
    t_emb = embed_texts(model, processor, t_texts)
    t_sim = cosine_matrix(t_emb, t_emb)
    print_matrix(t_sim, t_texts, t_texts, "text-text cosine (대각선 = 자기 자신 = 1.0)", mark_best=False)
    # 같은 종류(텍스트끼리)라 전부 0.8 이상으로 높게 나온다. 절대값이 아니라 '순서'를 보자:
    #   cat 기준 → kitten > dog > 고양이 사진 > bus  처럼 의미가 가까울수록 높다.
    save_heatmap(t_sim.numpy(), t_texts, t_texts, "text-text cosine similarity", "text_text.png")

    # 한국어 문장으로 이미지 검색도 해보자
    ko = ["버스", "넥타이를 맨 남자", "고양이 두 마리"]
    ko_sim = cosine_matrix(image_emb, embed_texts(model, processor, ko))
    print_matrix(ko_sim, image_names, ko, "한국어 문장 ↔ 이미지 cosine")

    # -----------------------------------------------------------------------
    # 실험 4. 이미지 ↔ 이미지 유사도
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("실험 4. 이미지 ↔ 이미지 코사인 유사도")
    print("=" * 70)
    # 좌우 반전한 이미지를 하나 추가: 픽셀은 완전히 달라졌지만 의미는 같다 → 유사도가 매우 높을 것
    flipped = images[0].transpose(Image.FLIP_LEFT_RIGHT)
    i_names = image_names + [f"{image_names[0]} (flip)"]
    i_emb = embed_images(model, processor, images + [flipped])
    i_sim = cosine_matrix(i_emb, i_emb)
    print_matrix(i_sim, i_names, i_names, "image-image cosine", mark_best=False)
    print("  → 반전 이미지와 원본의 유사도가 다른 이미지들보다 훨씬 높은지 확인하자.")
    print("    이것이 '이미지 검색(유사 이미지 찾기)'의 원리다.")
    save_heatmap(i_sim.numpy(), i_names, i_names, "image-image cosine similarity", "image_image.png")

    # -----------------------------------------------------------------------
    # 실습 과제 아이디어
    #   1. --texts 로 "a photo of a bus" 와 "bus" 를 함께 넣어보자. 프롬프트 형태가 점수를 바꾸나?
    #   2. "a photo of a red bus" vs "a photo of a blue bus" — 색깔을 구분하나?
    #   3. "a photo of a man to the left of a bus" 처럼 위치 관계는 이해할까?
    #      (CLIP 계열은 '무엇이 있나'는 잘 알지만 '어떻게 배치됐나'는 약한 편이다)
    #   4. 내 사진 폴더의 이미지들을 --images 로 넣고, 가장 비슷한 사진 쌍을 찾아보자.
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
