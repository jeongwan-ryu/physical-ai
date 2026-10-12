"""
YOLOE 실습 코드 — 세 가지 프롬프트 모드
=========================================

YOLOE(Real-Time Seeing Anything)는 "학습 때 보지 못한 물체"도 찾아내는
오픈 어휘(open-vocabulary) 객체 탐지 + 분할(segmentation) 모델이다.
기존 YOLO는 학습한 80개 클래스(COCO)만 찾을 수 있지만,
YOLOE는 "무엇을 찾을지"를 실행 시점에 알려줄 수 있다.

알려주는 방법(프롬프트)이 세 가지다:

  1) 텍스트 프롬프트 (Text Prompt)   : "person", "bus" 처럼 단어로 지시
     - 내부 기법: RepRTA (Re-parameterizable Region-Text Alignment)
     - 텍스트 인코더(MobileCLIP)로 단어를 임베딩 벡터로 바꾸고,
       이 벡터를 분류 헤드의 가중치처럼 사용한다.
     - 재파라미터화 덕분에 클래스를 한 번 설정하면 추론 시 추가 비용이 거의 없다.

  2) 이미지(시각) 프롬프트 (Visual Prompt) : 이미지 위에 박스로 "이런 거" 라고 지시
     - 내부 기법: SAVPE (Semantic-Activated Visual Prompt Encoder)
     - 말로 설명하기 어려운 물체(특정 부품, 특이한 모양 등)에 유용하다.

  3) 프롬프트 프리 (Prompt-Free)      : 아무 지시 없이 보이는 건 전부 찾기
     - 내부 기법: LRPC (Lazy Region-Prompt Contrast)
     - 약 4,585개의 내장 어휘 중에서 각 영역에 맞는 이름을 스스로 고른다.
     - 별도의 거대 언어모델 없이 동작하므로 빠르다.

설치:
    pip install -U ultralytics

실행 예:
    python yoloe_practice.py --mode text
    python yoloe_practice.py --mode visual
    python yoloe_practice.py --mode free
    python yoloe_practice.py --mode all              # 세 모드 모두
    python yoloe_practice.py --mode text --source my.jpg --classes cup laptop
    python yoloe_practice.py --mode text --source a.jpg b.jpg --classes tie   # 여러 이미지
    python yoloe_practice.py --mode text --source 0   # 웹캠 실시간

결과 이미지는 이 폴더의 runs/<모드>/ 아래에 저장된다.
가중치(.pt)와 예제 이미지는 처음 실행 시 이 폴더에 자동으로 다운로드된다.
(어느 위치에서 실행하든 스크립트가 자기 폴더로 이동해서 동작한다.)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLOE
from ultralytics.utils.checks import check_file

# 실습용 기본 이미지 (Ultralytics 공식 예제 이미지: 버스와 사람들이 찍힌 사진)
DEFAULT_SOURCE = "https://ultralytics.com/images/bus.jpg"

# 텍스트 모드 기본 이미지: bus.jpg + zidane.jpg (축구 감독과 선수, 넥타이가 보이는 사진)
#   predict() 에 리스트를 넘기면 여러 이미지를 한 번에 처리한다.
#   set_classes 로 정한 클래스는 모든 이미지에 똑같이 적용된다.
DEFAULT_TEXT_SOURCES = [
    DEFAULT_SOURCE,
    "https://ultralytics.com/images/zidane.jpg",
]

# 모델 크기: s(small) / m(medium) / l(large)
#   - 숫자 11 은 YOLO11 백본 기반이라는 뜻, "-seg" 는 분할 마스크까지 출력한다는 뜻
#   - 작을수록 빠르고, 클수록 정확하다. 실습은 s 로 충분하다.
#   - 최신 ultralytics 라면 "yoloe-26s-seg.pt" 같은 YOLO26 기반 모델도 쓸 수 있다.
MODEL_SIZE = "s"

# 이 스크립트가 있는 폴더. 가중치·이미지 다운로드와 결과 저장을 모두 여기서 한다.
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

# 결과 저장 폴더 (이 파일 옆의 runs). 상대경로를 주면 ultralytics 8.4+ 는
# runs/segment/ 아래에 한 번 더 중첩해 저장하므로 절대경로로 지정한다.
SAVE_DIR = str(HERE / "runs")


# ---------------------------------------------------------------------------
# 1) 텍스트 프롬프트 모드
# ---------------------------------------------------------------------------
def run_text_prompt(source, class_names, conf=0.25):
    """찾고 싶은 물체의 '이름'을 텍스트로 주고 탐지한다."""
    print(f"\n[1] 텍스트 프롬프트 모드 — 찾을 클래스: {class_names}")

    # 일반(프롬프트 가능한) YOLOE 모델 로드
    model = YOLOE(f"yoloe-11{MODEL_SIZE}-seg.pt")

    # get_text_pe(): 텍스트 → 임베딩(Prompt Embedding) 변환
    #   내부적으로 MobileCLIP 텍스트 인코더를 사용한다 (처음 실행 시 자동 다운로드).
    text_pe = model.get_text_pe(class_names)

    # set_classes(): 모델의 분류 헤드를 이 클래스들로 "재구성"한다.
    #   이후 model.predict() 는 오직 이 클래스들만 찾는다.
    #   결과의 클래스 번호 0, 1, 2... 는 class_names 의 순서와 같다.
    model.set_classes(class_names, text_pe)

    # 추론. conf 는 신뢰도 임계값 (낮추면 더 많이, 높이면 확실한 것만 검출)
    results = model.predict(
        source,
        conf=conf,
        save=True,                 # 박스/마스크가 그려진 이미지 저장
        project=SAVE_DIR,
        name="text",
        exist_ok=True,
    )
    print_results(results)

    # 팁: 클래스를 고정한 모델을 저장해두면 다음에는 텍스트 인코더 없이 바로 쓸 수 있다.
    #   model.save("yoloe_custom_classes.pt")
    return results


# ---------------------------------------------------------------------------
# 2) 이미지(시각) 프롬프트 모드
# ---------------------------------------------------------------------------
def run_visual_prompt(source, refer_image=None, conf=0.25):
    """이미지 위의 '예시 박스'를 주고, 그와 비슷한 물체를 찾는다."""
    print("\n[2] 이미지(시각) 프롬프트 모드")

    # 시각 프롬프트 전용 Predictor. (분할 모델이므로 SegPredictor 사용)
    #   탐지 전용 모델이라면 YOLOEVPDetectPredictor 를 쓴다.
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    model = YOLOE(f"yoloe-11{MODEL_SIZE}-seg.pt")

    # visual_prompts: "이 박스 안에 있는 게 내가 찾고 싶은 물체야" 라는 예시
    #   - bboxes: [x1, y1, x2, y2] 픽셀 좌표 (왼쪽 위, 오른쪽 아래)
    #   - cls   : 각 박스가 몇 번 클래스인지 (0부터 시작하는 정수)
    #             같은 번호를 여러 박스에 주면 "같은 종류의 예시 여러 개"가 된다.
    # 아래 좌표는 bus.jpg 기준:
    #   박스 0 → 왼쪽에 서 있는 사람 전체
    #   박스 1 → 사람이 쓴 안경 부분 (작은 물체)
    visual_prompts = dict(
        bboxes=np.array(
            [
                [221.52, 405.8, 344.98, 857.54],  # 사람
                [120, 425, 160, 445],             # 안경
            ]
        ),
        cls=np.array([0, 1]),
    )

    # 두 가지 사용법이 있다:
    #   (a) refer_image 없음 → 프롬프트 박스를 그린 바로 그 이미지(source)에서 비슷한 물체 탐지
    #   (b) refer_image 있음 → refer_image 에서 박스로 예시를 지정하고,
    #                          그 "개념"을 다른 이미지(source)에 적용해 탐지
    #       예) 부품 사진 한 장에 박스를 쳐두고, 공장 라인 영상 전체에서 그 부품 찾기
    results = model.predict(
        source,
        refer_image=refer_image,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        save=True,
        project=SAVE_DIR,
        name="visual",
        exist_ok=True,
    )
    # 시각 프롬프트는 이름이 없으므로 클래스 이름은 object0, object1 처럼 표시된다.
    print_results(results)
    return results


# ---------------------------------------------------------------------------
# 3) 프롬프트 프리 모드
# ---------------------------------------------------------------------------
def run_prompt_free(source, conf=0.25):
    """아무 지시 없이, 내장 어휘(약 4,585개)로 보이는 물체를 전부 찾는다."""
    print("\n[3] 프롬프트 프리 모드")

    # 파일 이름 끝의 "-pf" = prompt-free 전용 가중치
    #   내장 어휘 임베딩이 이미 모델 안에 들어 있어서 set_classes 가 필요 없다.
    model = YOLOE(f"yoloe-11{MODEL_SIZE}-seg-pf.pt")

    results = model.predict(
        source,
        conf=conf,
        save=True,
        project=SAVE_DIR,
        name="free",
        exist_ok=True,
    )
    # 텍스트 모드보다 훨씬 다양한(때로는 엉뚱한) 이름이 나오는 걸 관찰해보자.
    print_results(results)
    return results


# ---------------------------------------------------------------------------
# 결과 해석 도우미
# ---------------------------------------------------------------------------
def print_results(results):
    """Results 객체에서 탐지 정보를 꺼내 출력한다. (결과 구조 공부용)"""
    for r in results:
        # r.boxes     : 탐지된 박스들
        #   .xyxy     : [N, 4] 박스 좌표 (x1, y1, x2, y2)
        #   .conf     : [N]    신뢰도
        #   .cls      : [N]    클래스 번호
        # r.masks     : 분할 마스크 (seg 모델일 때만), r.masks.data → [N, H, W]
        # r.names     : {클래스 번호: 이름} 딕셔너리
        print(f"  이미지: {r.path}")
        if r.boxes is None or len(r.boxes) == 0:
            print("  → 탐지된 물체 없음 (conf 를 낮춰보자)")
            continue
        for box in r.boxes:
            cls_id = int(box.cls)
            name = r.names[cls_id]
            score = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  - {name:<15} conf={score:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        if r.masks is not None:
            print(f"  마스크 개수: {len(r.masks)}  (shape={tuple(r.masks.data.shape)})")
        print(f"  저장 위치: {r.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLOE 세 가지 프롬프트 모드 실습")
    parser.add_argument("--mode", choices=["text", "visual", "free", "all"], default="all")
    parser.add_argument(
        "--source", nargs="+", default=None,
        help="이미지/영상 경로, URL, 또는 웹캠 번호(0). 여러 개를 띄어쓰기로 나열 가능",
    )
    parser.add_argument(
        "--classes", nargs="+", default=["person", "bus", "glasses", "tie"],
        help="텍스트 모드에서 찾을 클래스 이름들",
    )
    parser.add_argument("--refer", default=None, help="시각 모드: 예시 박스를 지정할 참조 이미지 (생략 시 source 사용)")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    args = parser.parse_args()

    def to_source(items):
        # 웹캠 번호("0")는 정수로 바꿔줘야 한다. 한 개면 리스트가 아닌 단일 값으로 넘긴다.
        # URL 은 check_file() 로 먼저 내려받아 두면 결과가 원래 파일 이름(bus.jpg 등)으로 저장된다.
        items = [int(x) if x.isdigit() else check_file(x) if x.startswith("http") else user_path(x) for x in items]
        return items[0] if len(items) == 1 else items

    if args.mode in ("text", "all"):
        run_text_prompt(to_source(args.source or DEFAULT_TEXT_SOURCES), args.classes, args.conf)
    if args.mode in ("visual", "all"):
        # 시각 프롬프트의 박스 좌표는 특정 이미지 한 장 기준이므로 첫 번째 이미지만 사용한다.
        refer = user_path(args.refer) if args.refer else None
        run_visual_prompt(to_source((args.source or [DEFAULT_SOURCE])[:1]), refer, args.conf)
    if args.mode in ("free", "all"):
        run_prompt_free(to_source(args.source or [DEFAULT_SOURCE]), args.conf)

    # -----------------------------------------------------------------------
    # 실습 과제 아이디어
    #   1. 텍스트 모드에서 "red car", "person wearing hat" 처럼 수식어를 붙이면 어떻게 될까?
    #   2. 시각 모드에서 박스 좌표를 바꿔 다른 물체를 예시로 줘보자.
    #      (좌표는 그림판 등에서 마우스를 올려 확인 가능)
    #   3. 같은 이미지를 세 모드로 돌려 결과를 비교해보자.
    #      - 텍스트: 지정한 것만 정확히
    #      - 시각  : 이름 없는 물체도 예시로 찾기
    #      - 프리  : 지정 없이 넓게, 대신 이름이 부정확할 수 있음
    #   4. MODEL_SIZE 를 s → l 로 바꿔 속도/정확도 차이를 확인해보자.
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
