r"""
스터디 노트(siglip_image_encoder.md)에 들어가는 그림을 만드는 스크립트.

실제 모델(google/siglip2-base-patch16-224)과 예제 이미지(bus.jpg)를 써서
패치 분할 → 패치 임베딩 → self-attention → MAP 풀링 → 텍스트와 비교 과정을 그림으로 저장한다.

실행 (test 폴더에서):
    py siglip2\docs\make_figures.py
결과: siglip2/docs/images/*.png
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from transformers import AutoModel, AutoProcessor

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
transformers.logging.set_verbosity_error()
plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)
IMG_PATH = HERE.parent / "bus.jpg"
MODEL_ID = "google/siglip2-base-patch16-224"
P = 16            # 패치 크기
G = 224 // P      # 한 변의 패치 수 = 14  → 총 196개
QUERY = (7, 9)    # 실험용 기준 패치 (행, 열) — 버스 몸통(로고) 부분

TEXTS = ["a photo of a bus", "a photo of two cats", "a photo of a man", "a photo of a street"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", OUT / name)


def box(ax, x, y, w, h, text, color, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.4, color="#333"))


# ---------------------------------------------------------------- 그림 1: 전체 흐름
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [
        ("이미지\n224×224×3", "#e8eef7"),
        ("① 패치 분할\n16×16 → 196개", "#dbe9ff"),
        ("② 패치 임베딩\n선형변환 + 위치정보\n→ 196×768", "#d6f0e0"),
        ("③ Transformer\n인코더 ×12\n(self-attention)", "#fff1cc"),
        ("④ MAP 풀링\n196개 → 1개\n768차원 벡터", "#ffe0e0"),
        ("텍스트 벡터와\n비교 (sigmoid)", "#eee3ff"),
    ]
    w, h, gap = 1.85, 1.8, 0.32
    for i, (t, c) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, 1.05, w, h, t, c, fs=10)
        if i:
            arrow(ax, x - gap + 0.03, 1.95, x - 0.03, 1.95)
    ax.text(6.5, 0.45, "①~④ = SigLIP 이미지 인코더 (ViT-B/16 기준)", ha="center", fontsize=11, color="#555")
    ax.plot([2.2, 10.9], [0.85, 0.85], color="#999", lw=1)
    save(fig, "fig1_pipeline.png")


# ---------------------------------------------------------------- 그림 2: 패치 분할
def fig_patches(img224):
    arr = np.asarray(img224)
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([0.0, 0.0, 0.48, 1.0])
    ax.imshow(arr)
    for k in range(G + 1):
        ax.axhline(k * P - 0.5, color="white", lw=0.8)
        ax.axvline(k * P - 0.5, color="white", lw=0.8)
    picks = [(2, 3), QUERY, (11, 7), (1, 11)]
    colors = ["#ff4d4d", "#ffb000", "#22c55e", "#3b82f6"]
    for (r, c), col in zip(picks, colors):
        ax.add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, fill=False, ec=col, lw=2.5))
    ax.set_title(f"224×224 이미지를 16×16 패치로 자르면 {G}×{G} = {G*G}개", fontsize=12)
    ax.axis("off")

    # 오른쪽: 잘라낸 패치들을 일렬로 (= 토큰 시퀀스)
    for i, ((r, c), col) in enumerate(zip(picks, colors)):
        a = fig.add_axes([0.55 + i * 0.11, 0.45, 0.09, 0.2])
        a.imshow(arr[r * P:(r + 1) * P, c * P:(c + 1) * P], interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3)
        a.set_title(f"패치 #{r*G+c}", fontsize=9)
    fig.text(0.76, 0.75, "패치 하나하나가 '단어'처럼 취급되는 토큰", ha="center", fontsize=12)
    fig.text(0.76, 0.33, "순서: 왼쪽 위 → 오른쪽 아래 (#0 ~ #195)\n각 패치 = 16×16×3(RGB) = 768개의 숫자",
             ha="center", fontsize=10.5, color="#444")
    save(fig, "fig2_patches.png")


# ---------------------------------------------------------------- 그림 3: 패치 임베딩
def fig_patch_embedding(img224, emb_patch, emb_pos):
    arr = np.asarray(img224)
    r, c = QUERY
    idx = r * G + c
    patch = arr[r * P:(r + 1) * P, c * P:(c + 1) * P]
    raw = patch.reshape(-1).astype(np.float32) / 255.0                 # 768개의 픽셀값
    pe = emb_patch[idx]                                                # Conv(=선형변환) 결과
    pos = emb_pos[idx]                                                 # 위치 임베딩
    tot = pe + pos

    fig = plt.figure(figsize=(12, 6.2))
    a0 = fig.add_axes([0.02, 0.55, 0.14, 0.33])
    a0.imshow(patch, interpolation="nearest"); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f"패치 #{idx}\n(16×16×3)", fontsize=10)

    rows = [
        (0.80, raw, "펼친 픽셀값 768개 (0~1)", "gray"),
        (0.60, pe, "학습된 선형변환 후 768차원 (Conv 16×16, stride 16)", "coolwarm"),
        (0.40, pos, "+ 위치 임베딩 768차원 (#%d 자리에 해당하는 학습된 값)" % idx, "PiYG"),
        (0.20, tot, "= 패치 임베딩 (Transformer 입력)", "coolwarm"),
    ]
    for y, v, title, cmap in rows:
        a = fig.add_axes([0.22, y, 0.75, 0.07])
        lim = np.abs(v).max()
        a.imshow(v[None, :], aspect="auto", cmap=cmap,
                 vmin=(0 if cmap == "gray" else -lim), vmax=(1 if cmap == "gray" else lim))
        a.set_yticks([]); a.set_xticks([0, 255, 511, 767])
        a.tick_params(labelsize=8)
        a.set_title(title, fontsize=10.5, loc="left")
    fig.text(0.59, 0.05, "픽셀 → 벡터 변환은 '사람이 정한 규칙'이 아니라 학습으로 정해진 가중치(768×768)로 계산된다.\n"
             "위치 임베딩이 없으면 Transformer는 패치가 이미지의 어디에 있었는지 알 수 없다.",
             ha="center", fontsize=10, color="#444")
    save(fig, "fig3_patch_embedding.png")


# ---------------------------------------------------------------- 그림 4: self-attention
def fig_attention(img224, attentions):
    arr = np.asarray(img224)
    r, c = QUERY
    idx = r * G + c
    layers = [1, 3, 8]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    axes[0].imshow(arr)
    axes[0].add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, fill=False, ec="red", lw=2.5))
    axes[0].set_title(f"기준 패치 #{idx} (빨간 칸)", fontsize=11)
    for ax, L in zip(axes[1:], layers):
        att = attentions[L][0].mean(0)[idx].clone()    # 헤드 평균: 기준 패치가 196개 패치를 각각 얼마나 보는가
        att[idx] = 0                                   # 자기 자신은 빼고 표시
        m = att.reshape(G, G).numpy()
        m = np.kron(m / m.max(), np.ones((P, P)))      # 224×224로 확대
        ax.imshow(arr); ax.imshow(m, cmap="jet", alpha=0.5)
        ax.add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, fill=False, ec="white", lw=2))
        ax.set_title(f"{L+1}번째 층", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Self-attention: 기준 패치가 다른 패치들을 얼마나 참고하는가 (실제 모델, 12개 헤드 평균, 자기 자신 제외)",
                 fontsize=12, y=1.02)
    save(fig, "fig4_self_attention.png")


# ---------------------------------------------------------------- 그림 4b: attention sink
def fig_sink(img224, attentions, hidden_states):
    arr = np.asarray(img224)
    L = 8
    norms = hidden_states[L + 1][0].norm(dim=-1)       # 9번째 층 출력의 토큰별 벡터 길이
    recv = attentions[L][0].mean(0).mean(0)            # 모든 패치가 각 패치를 평균적으로 얼마나 보는가
    sink = int(norms.argmax())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    for ax, v, t in [(axes[0], norms, f"토큰별 벡터 길이 (중앙값 {norms.median():.0f}, 최대 {norms.max():.0f})"),
                     (axes[1], recv, "모든 패치가 받는 attention 평균")]:
        m = v.reshape(G, G).numpy()
        ax.imshow(arr); ax.imshow(np.kron(m / m.max(), np.ones((P, P))), cmap="jet", alpha=0.5)
        ax.add_patch(Rectangle((sink % G * P - 0.5, sink // G * P - 0.5), P, P, fill=False, ec="white", lw=2))
        ax.set_title(t, fontsize=11); ax.axis("off")
    fig.suptitle(f"{L+1}번째 층: 배경 패치 #{sink} 하나에 attention이 몰린다 (attention sink / 레지스터 토큰 현상)",
                 fontsize=12, y=1.0)
    save(fig, "fig4b_attention_sink.png")


# ---------------------------------------------------------------- 그림 5: MAP 풀링
def fig_map_pooling(img224, map_weights):
    arr = np.asarray(img224)
    m = map_weights.reshape(G, G).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(arr); axes[0].set_title("입력 이미지", fontsize=11)
    axes[1].imshow(arr)
    axes[1].imshow(np.kron(m / m.max(), np.ones((P, P))), cmap="jet", alpha=0.5)
    axes[1].set_title("MAP 헤드가 각 패치에 준 가중치", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("④ 학습된 query 1개가 196개 패치를 attention으로 가중합 → 이미지 벡터 1개 (768차원)",
                 fontsize=12, y=1.0)
    save(fig, "fig5_map_pooling.png")


# ---------------------------------------------------------------- 그림 6: 텍스트와 비교
def fig_compare(cos, probs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    y = np.arange(len(TEXTS))[::-1]
    axes[0].barh(y, cos, color=["#3b82f6" if i == cos.argmax() else "#bbb" for i in range(len(cos))])
    axes[0].set_title("코사인 유사도  (이미지 벡터 · 텍스트 벡터)", fontsize=11)
    axes[1].barh(y, probs * 100, color=["#ef4444" if i == probs.argmax() else "#bbb" for i in range(len(probs))])
    axes[1].set_title("sigmoid(cos × t + b)  → '맞는 쌍일 확률' (%)", fontsize=11)
    for ax, vals, fmt in [(axes[0], cos, "{:+.3f}"), (axes[1], probs * 100, "{:.3f}%")]:
        ax.set_yticks(y); ax.set_yticklabels(TEXTS)
        for yy, v in zip(y, vals):
            ax.text(max(v, 0), yy, "  " + fmt.format(v), va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_yticklabels([])
    fig.suptitle("bus.jpg 의 이미지 벡터를 문장 4개와 비교 (각 쌍을 독립적으로 판정 → 확률 합이 1이 아님)",
                 fontsize=12, y=1.04)
    save(fig, "fig6_compare_text.png")


@torch.no_grad()
def main():
    img = Image.open(IMG_PATH).convert("RGB")
    img224 = img.resize((224, 224), Image.BICUBIC)     # 프로세서도 크롭 없이 224×224로 리사이즈한다

    model = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vm = model.vision_model

    px = processor(images=img, return_tensors="pt")["pixel_values"]
    emb_patch = vm.embeddings.patch_embedding(px).flatten(2).transpose(1, 2)[0].numpy()   # 196×768
    emb_pos = vm.embeddings.position_embedding.weight.numpy()                            # 196×768

    out = vm(pixel_values=px, output_attentions=True, output_hidden_states=True)
    hidden = out.last_hidden_state                                                       # 1×196×768

    # MAP 헤드 attention 가중치 직접 계산 (probe = 학습된 query 1개)
    head = vm.head
    probe = head.probe.repeat(hidden.shape[0], 1, 1)
    _, w = head.attention(probe, hidden, hidden, need_weights=True)
    map_weights = w[0, 0]                                                                # 196

    t = processor(text=TEXTS, padding="max_length", max_length=64, return_tensors="pt")
    te = model.get_text_features(**t)
    te = F.normalize(te if isinstance(te, torch.Tensor) else te.pooler_output, dim=-1)
    ie = F.normalize(out.pooler_output, dim=-1)
    cos = (ie @ te.T)[0]
    probs = torch.sigmoid(cos * model.logit_scale.exp() + model.logit_bias)

    fig_pipeline()
    fig_patches(img224)
    fig_patch_embedding(img224, emb_patch, emb_pos)
    fig_attention(img224, out.attentions)
    fig_sink(img224, out.attentions, out.hidden_states)
    fig_map_pooling(img224, map_weights)
    fig_compare(cos.numpy(), probs.numpy())
    print("cos:", [round(v, 3) for v in cos.tolist()], "prob:", [round(v, 3) for v in probs.tolist()])


if __name__ == "__main__":
    main()
