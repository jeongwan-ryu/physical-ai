"""
SAM 2 실습 코드 — "가리키면 잘라준다"
======================================

SAM(Segment Anything Model)은 Meta가 만든 '범용 분할(segmentation)' 모델이다.
SAM 2는 그 후속작으로, 이미지뿐 아니라 '영상'에서도 물체를 따라가며 분할한다.

YOLO/YOLOE 와의 차이를 먼저 잡고 가자:
  - YOLOE : "무엇(what)"을 찾는다  → 클래스 이름 + 박스 (+ 마스크)
  - SAM 2 : "어디(where)"를 자른다 → 이름은 모른다. 가리킨 곳의 '정확한 경계'만 준다.
  그래서 둘을 합치면(YOLOE로 찾고 SAM으로 자르기) 이름 + 정밀 마스크를 얻는다. (--mode yoloe)

SAM 2 의 구조 (이걸 알면 아래 코드가 이해된다):
  ┌──────────────┐   이미지 임베딩    ┌──────────────┐
  │ Image Encoder│ ───────────────▶ │ Mask Decoder │ ──▶ 마스크
  │  (무겁다)     │                  │  (가볍다)     │
  └──────────────┘                  └──────▲───────┘
                                           │
                               ┌───────────┴──────────┐
                               │ Prompt Encoder        │
                               │ 점(+/-), 박스, 마스크  │
                               └──────────────────────┘
  - 이미지 인코더는 이미지당 '한 번만' 돌리면 된다 (느림, 대부분의 연산이 여기)
  - 프롬프트를 바꿔가며 디코더만 다시 돌리면 거의 즉시 새 마스크가 나온다 (빠름)
  - 영상에서는 여기에 'Memory Bank'가 붙어서 이전 프레임의 마스크를 기억하고
    다음 프레임에서 같은 물체를 계속 추적한다.

프롬프트 종류:
  - 양성 점 (label=1) : "여기는 물체야"
  - 음성 점 (label=0) : "여기는 물체가 아니야" → 마스크가 너무 넓을 때 깎아내는 용도
  - 박스             : "이 박스 안의 물체" → 가장 안정적인 프롬프트
  - 프롬프트 없음      : 이미지 전체를 격자 점으로 찍어 '모든 것'을 분할 (느림)

실행 예:
    py sam2_practice.py --mode point        # 점 하나로 분할
    py sam2_practice.py --mode points       # 양성/음성 점 조합 (점 하나와 결과 비교)
    py sam2_practice.py --mode box          # 박스로 분할
    py sam2_practice.py --mode multi        # 여러 물체를 한 번에
    py sam2_practice.py --mode everything   # 프롬프트 없이 전부 분할
    py sam2_practice.py --mode yoloe --classes person bus   # YOLOE + SAM2 조합
    py sam2_practice.py --mode click        # ★ 마우스로 직접 찍어보는 대화형 모드
    py sam2_practice.py --mode video --source my_video.mp4 --point 500 300  # 영상 추적
    py sam2_practice.py --mode all          # click/video 제외 전부

    다른 이미지:  --source 내사진.jpg  (point/box 좌표는 --point / --box 로 바꿔줄 것)

결과는 이 폴더의 runs/<모드>.jpg 로 저장된다.
가중치와 예제 이미지는 이 폴더에 다운로드된다 (어디서 실행하든 스크립트 폴더로 이동해서 동작).
yoloe 모드는 옆 폴더(../yoloe)의 YOLOE 가중치와 텍스트 인코더를 그대로 빌려 쓴다. 프롬프트(점/박스)도 그림에 함께 표시된다.
  초록 점 = 양성(물체), 빨간 점 = 음성(배경), 노란 박스 = 박스 프롬프트
"""

import argparse
import contextlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.utils.checks import check_file

# 모델 크기: t(tiny) / s(small) / b(base) / l(large)
#   "2.1" 은 SAM 2 의 개선판 가중치. CPU 에서는 tiny 가 무난하다.
#   GPU 가 있다면 b 나 l 로 바꿔 경계가 얼마나 정교해지는지 비교해보자.
SAM_MODEL = "sam2.1_t.pt"

DEFAULT_SOURCE = "https://ultralytics.com/images/bus.jpg"
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

# yoloe 모드에서 사용할 YOLOE 실습 폴더 (가중치와 600MB 짜리 텍스트 인코더가 여기 있다)
YOLOE_DIR = HERE.parent / "yoloe"

# bus.jpg (810x1080) 기준 예시 좌표
PERSON_POINT = [283, 600]                 # 왼쪽에서 두 번째 사람의 몸통
PERSON_BOX = [221, 405, 345, 857]         # 같은 사람을 감싸는 박스 [x1, y1, x2, y2]
BUS_POINT = [600, 300]                    # 버스 차체


# ---------------------------------------------------------------------------
# 시각화 / 출력 도우미
# ---------------------------------------------------------------------------
def save_result(result, name, points=None, labels=None, box=None, title=None):
    """마스크가 칠해진 이미지 위에 프롬프트(점/박스)까지 그려서 저장한다."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # result.plot(): 마스크(+박스)를 그린 BGR 이미지(numpy)를 돌려준다.
    #   SAM 은 클래스 이름이 없으므로 라벨은 0, 1, 2 ... 번호로 표시된다.
    img = result.plot(labels=False, boxes=False)

    if box is not None:
        for b in np.array(box).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if points is not None:
        pts = np.array(points).reshape(-1, 2)
        lbs = np.array(labels if labels is not None else [1] * len(pts)).reshape(-1)
        for (x, y), lb in zip(pts, lbs):
            color = (0, 255, 0) if lb == 1 else (0, 0, 255)  # BGR: 초록=양성, 빨강=음성
            cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), -1)
            cv2.circle(img, (int(x), int(y)), 9, color, -1)
    if title:
        cv2.putText(img, title, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(img, title, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    path = SAVE_DIR / f"{name}.jpg"
    cv2.imwrite(str(path), img)
    print(f"  저장: {path}")
    return img


def describe(result):
    """마스크 정보 출력. result.masks.data 는 [N, H, W] 의 0/1 텐서다."""
    if result.masks is None:
        print("  → 마스크 없음")
        return
    masks = result.masks.data.cpu().numpy()
    h, w = masks.shape[1:]
    print(f"  마스크 {len(masks)}개 (각 {w}x{h})")
    for i, m in enumerate(masks[:10]):
        area = int(m.sum())
        print(f"   - #{i}: 면적 {area:,} px ({area / (h * w) * 100:.1f}% of image)")
    if len(masks) > 10:
        print(f"   ... 외 {len(masks) - 10}개")


def run(model, source, **prompts):
    """SAM 추론 + 시간 측정. prompts 는 points/labels/bboxes 중 필요한 것."""
    t = time.perf_counter()
    result = model(source, verbose=False, **prompts)[0]
    print(f"  추론 시간: {time.perf_counter() - t:.2f}s")
    describe(result)
    return result


# ---------------------------------------------------------------------------
# 1) 점 하나
# ---------------------------------------------------------------------------
def mode_point(model, source, point):
    print(f"\n[point] 양성 점 하나: {point}")
    # points=[x, y], labels=[1]  → "이 점이 포함된 물체를 잘라줘"
    # 점 하나는 모호하다. 옷을 찍으면 '옷'일 수도 '사람 전체'일 수도 있다.
    # SAM 은 내부적으로 여러 후보를 만든 뒤 가장 점수가 높은 것을 고른다.
    r = run(model, source, points=[point], labels=[1])
    save_result(r, "point", points=[point], labels=[1], title="1 positive point")


# ---------------------------------------------------------------------------
# 2) 양성 + 음성 점 조합 (한 물체에 여러 점)
# ---------------------------------------------------------------------------
def mode_points(model, source):
    print("\n[points] 한 물체에 양성/음성 점 여러 개")
    # 주의: 괄호 깊이가 의미를 바꾼다!
    #   points=[[x1,y1],[x2,y2]],   labels=[1,1]     → 점마다 '서로 다른 물체' 2개
    #   points=[[[x1,y1],[x2,y2]]], labels=[[1,0]]   → '한 물체'에 점 2개 (여기서 쓰는 방식)
    #
    # 실험: 버스 차체에 양성 점 하나만 찍으면 버스가 잡힌다.
    #       여기에 버스 앞에 선 사람 위에 음성 점을 추가하면 → 사람 부분이 마스크에서 빠진다.
    pts = [BUS_POINT, PERSON_POINT]

    print(" (a) 버스에 양성 점 하나만")
    r1 = run(model, source, points=[BUS_POINT], labels=[1])
    save_result(r1, "points_a_positive_only", points=[BUS_POINT], labels=[1], title="positive only")

    print(" (b) + 사람 위에 음성 점")
    r2 = run(model, source, points=[pts], labels=[[1, 0]])
    save_result(r2, "points_b_with_negative", points=pts, labels=[1, 0], title="positive + negative")
    print("  → 두 이미지를 나란히 열어 사람 영역이 빠졌는지 비교해보자.")


# ---------------------------------------------------------------------------
# 3) 박스
# ---------------------------------------------------------------------------
def mode_box(model, source, box):
    print(f"\n[box] 박스 프롬프트: {box}")
    # 박스는 '대략 이 영역'을 알려주므로 점보다 모호함이 적다.
    # 그래서 탐지기(YOLO 등)의 박스 → SAM 마스크 로 연결하는 파이프라인이 흔하다.
    r = run(model, source, bboxes=[box])
    save_result(r, "box", box=box, title="box prompt")


# ---------------------------------------------------------------------------
# 4) 여러 물체를 한 번에
# ---------------------------------------------------------------------------
def mode_multi(model, source):
    print("\n[multi] 점 여러 개 = 물체 여러 개")
    # 괄호 한 단계: 점 하나하나가 독립된 프롬프트 → 물체마다 마스크 1개씩
    pts = [PERSON_POINT, BUS_POINT, [750, 760], [140, 650]]  # 사람, 버스, 오른쪽 사람, 왼쪽 사람
    r = run(model, source, points=pts, labels=[1] * len(pts))
    save_result(r, "multi", points=pts, labels=[1] * len(pts), title=f"{len(pts)} objects")


# ---------------------------------------------------------------------------
# 5) 프롬프트 없이 전부
# ---------------------------------------------------------------------------
def mode_everything(model, source):
    print("\n[everything] 프롬프트 없이 모든 것 분할 (CPU 에서는 수십 초 걸릴 수 있음)")
    # 프롬프트를 안 주면 이미지 위에 격자 모양으로 점을 잔뜩 찍고
    # 각 점마다 마스크를 만든 뒤 겹치는 것/품질 낮은 것을 걸러낸다.
    # → 이름은 없지만 '물체 같은 덩어리'를 전부 얻는다. 데이터 라벨링 보조에 많이 쓴다.
    r = run(model, source)
    save_result(r, "everything", title="segment everything")


# ---------------------------------------------------------------------------
# 6) YOLOE(무엇) + SAM2(어디) 조합
# ---------------------------------------------------------------------------
def mode_yoloe(model, source, classes):
    print(f"\n[yoloe] YOLOE 로 {classes} 를 찾고 → 그 박스를 SAM2 에 넣어 정밀 분할")
    from ultralytics import YOLOE

    # YOLOE 가중치(yoloe-11s-seg.pt)와 텍스트 인코더(mobileclip_blt.ts)는 현재 폴더에서 찾으므로,
    # 잠시 ../yoloe 폴더로 이동해서 로드한다. 이렇게 하면 큰 파일을 중복 다운로드하지 않는다.
    with contextlib.chdir(YOLOE_DIR):
        det = YOLOE("yoloe-11s-seg.pt")
        det.set_classes(classes, det.get_text_pe(classes))
    d = det.predict(source, conf=0.3, verbose=False)[0]
    if len(d.boxes) == 0:
        print("  → YOLOE 가 아무것도 찾지 못했다.")
        return
    boxes = d.boxes.xyxy.cpu().numpy()
    names = [d.names[int(c)] for c in d.boxes.cls]
    print(f"  YOLOE 탐지: {names}")

    # 박스 여러 개를 한 번에 넣으면 박스마다 마스크가 하나씩 나온다.
    r = run(model, source, bboxes=boxes.tolist())
    save_result(r, "yoloe_sam2", box=boxes, title="YOLOE box -> SAM2 mask")
    print("  → ../yoloe/runs/text 의 YOLOE 자체 마스크와 경계 품질을 비교해보자.")


# ---------------------------------------------------------------------------
# 7) ★ 대화형: 마우스로 클릭하며 체험
# ---------------------------------------------------------------------------
def mode_click(source):
    """
    조작법
      좌클릭  : 양성 점 추가 (물체)
      우클릭  : 음성 점 추가 (배경)
      드래그  : (좌클릭한 채로 끌기) 박스 프롬프트
      r       : 초기화
      s       : 현재 화면 저장
      q / ESC : 종료

    여기서는 SAM 모델 대신 'Predictor' 를 직접 쓴다.
      predictor.set_image(img) : 이미지 인코더를 '한 번만' 실행해 임베딩을 캐시
      predictor(points=...)    : 캐시된 임베딩 + 새 프롬프트로 디코더만 실행
    클릭할 때마다 걸리는 시간이 set_image 보다 훨씬 짧은 것을 터미널에서 확인해보자.
    이것이 SAM 이 '실시간 대화형 분할'을 할 수 있는 이유다.
    """
    from ultralytics.models.sam import SAM2Predictor

    img = cv2.imread(source)
    predictor = SAM2Predictor(overrides=dict(model=SAM_MODEL, save=False, verbose=False))

    t = time.perf_counter()
    predictor.set_image(img)
    print(f"\n[click] 이미지 인코딩(한 번만): {time.perf_counter() - t:.2f}s")
    print("  좌클릭=물체, 우클릭=배경, 드래그=박스, r=초기화, s=저장, q=종료")

    state = dict(points=[], labels=[], box=None, drag_start=None, view=img.copy())

    def redraw():
        view = img.copy()
        if state["points"] or state["box"] is not None:
            prompts = {}
            if state["points"]:
                # 모든 점을 '한 물체'의 프롬프트로 묶는다 (괄호 한 단계 더)
                prompts["points"] = [state["points"]]
                prompts["labels"] = [state["labels"]]
            if state["box"] is not None:
                prompts["bboxes"] = [state["box"]]
            t = time.perf_counter()
            r = predictor(**prompts)[0]
            dt = time.perf_counter() - t
            if r.masks is not None:
                m = r.masks.data[0].cpu().numpy().astype(bool)
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0])).astype(bool)
                overlay = view.copy()
                overlay[m] = (255, 144, 30)
                view = cv2.addWeighted(overlay, 0.5, view, 0.5, 0)
                # 마스크 외곽선
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(view, cnts, -1, (255, 255, 255), 2)
                print(f"  디코딩: {dt * 1000:.0f}ms | 점 {len(state['points'])}개 | 면적 {int(m.sum()):,}px")
        if state["box"] is not None:
            x1, y1, x2, y2 = map(int, state["box"])
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for (x, y), lb in zip(state["points"], state["labels"]):
            cv2.circle(view, (x, y), 8, (255, 255, 255), -1)
            cv2.circle(view, (x, y), 6, (0, 255, 0) if lb == 1 else (0, 0, 255), -1)
        state["view"] = view
        cv2.imshow("SAM2 click", view)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"]:
            x0, y0 = state["drag_start"]
            state["drag_start"] = None
            if abs(x - x0) > 10 and abs(y - y0) > 10:  # 끌었으면 박스
                state["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            else:                                      # 그냥 클릭이면 양성 점
                state["points"].append([x, y])
                state["labels"].append(1)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["points"].append([x, y])
            state["labels"].append(0)
            redraw()

    cv2.namedWindow("SAM2 click", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM2 click", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty("SAM2 click", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            state.update(points=[], labels=[], box=None)
            redraw()
        if key == ord("s"):
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVE_DIR / "click.jpg"
            cv2.imwrite(str(path), state["view"])
            print(f"  저장: {path}")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 8) 영상 추적
# ---------------------------------------------------------------------------
def mode_video(source, point):
    """
    첫 프레임에 점 하나만 찍으면, SAM 2 가 Memory Bank 로 이전 프레임들을 기억하면서
    이후 프레임 전체에서 그 물체를 계속 분할(추적)한다.
    YOLO 처럼 매 프레임 새로 탐지하는 게 아니라 '그 물체 하나'를 끝까지 따라간다는 점이 핵심.
    (CPU 에서는 느리다. 5~10초 정도의 짧은 영상으로 시작하자.)
    """
    from ultralytics.models.sam import SAM2VideoPredictor

    print(f"\n[video] 첫 프레임의 점 {point} 에 있는 물체를 영상 끝까지 추적")
    predictor = SAM2VideoPredictor(
        overrides=dict(conf=0.25, task="segment", mode="predict", model=SAM_MODEL,
                       save=True, project=str(SAVE_DIR), name="video", exist_ok=True)
    )
    # stream=True: 프레임을 하나씩 받아 처리 (메모리 절약)
    for i, r in enumerate(predictor(source=source, points=[point], labels=[1], stream=True)):
        if i % 30 == 0:
            n = 0 if r.masks is None else len(r.masks)
            print(f"  frame {i}: 마스크 {n}개")
    print(f"  저장 폴더: {SAVE_DIR / 'video'}")


def main():
    parser = argparse.ArgumentParser(description="SAM 2 실습")
    parser.add_argument(
        "--mode", default="all",
        choices=["point", "points", "box", "multi", "everything", "yoloe", "click", "video", "all"],
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="이미지(또는 video 모드에서는 영상) 경로/URL")
    parser.add_argument("--point", nargs=2, type=int, default=PERSON_POINT, metavar=("X", "Y"))
    parser.add_argument("--box", nargs=4, type=int, default=PERSON_BOX, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--classes", nargs="+", default=["person", "bus"], help="yoloe 모드에서 찾을 물체")
    args = parser.parse_args()

    source = check_file(args.source) if args.source.startswith("http") else user_path(args.source)

    if args.mode == "click":
        return mode_click(source)
    if args.mode == "video":
        return mode_video(source, args.point)

    # SAM 모델 로드 (처음 실행 시 가중치 자동 다운로드)
    model = SAM(SAM_MODEL)
    model.info()  # 파라미터 수 등 모델 정보 출력

    if args.mode in ("point", "all"):
        mode_point(model, source, args.point)
    if args.mode in ("points", "all"):
        mode_points(model, source)
    if args.mode in ("box", "all"):
        mode_box(model, source, args.box)
    if args.mode in ("multi", "all"):
        mode_multi(model, source)
    if args.mode in ("yoloe", "all"):
        mode_yoloe(model, source, args.classes)
    if args.mode in ("everything", "all"):
        mode_everything(model, source)

    # -----------------------------------------------------------------------
    # 실습 과제 아이디어
    #   1. --mode click 에서 사람의 '옷'만 클릭 → 옷만 잡히나, 사람 전체가 잡히나?
    #      음성 점을 추가해 원하는 범위로 다듬어보자.
    #   2. 같은 물체를 점 하나 / 박스로 각각 분할해 경계 품질을 비교해보자.
    #   3. SAM_MODEL 을 sam2.1_t → sam2.1_b 로 바꿔 속도와 경계 정밀도를 비교해보자.
    #   4. 휴대폰으로 5초짜리 영상을 찍어 --mode video 로 추적해보자.
    #      물체가 잠깐 가려졌다 다시 나타나도 따라가는지 관찰하자 (Memory Bank 의 역할).
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
