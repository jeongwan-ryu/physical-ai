"""
OCR 실습 코드 — 이미지 속 글자 읽기 (EasyOCR)
=============================================

OCR(Optical Character Recognition) = 이미지 속 글자를 텍스트 데이터로 바꾸는 기술.
로봇이 라벨·표지판·계기판을 읽거나, 문서를 디지털화할 때 쓴다.

EasyOCR 은 OCR 을 두 단계로 나눠서 처리한다:

  이미지 ──▶ ① 글자 검출 (Detection) ──▶ 글자 영역 박스들 ──▶ ② 글자 인식 (Recognition) ──▶ "텍스트"
             CRAFT 모델                                         CRNN 모델

  ① CRAFT (Character Region Awareness For Text detection)
     - 이미지의 각 픽셀이 '글자의 중심'일 확률과 '글자 사이 연결부'일 확률을 히트맵으로 예측한다.
     - 이 히트맵을 이어 붙여 단어/줄 단위 박스를 만든다.
     - "어디에 글자가 있나?" 만 알고, 무슨 글자인지는 모른다.

  ② CRNN (Convolutional Recurrent Neural Network)
     - 잘라낸 글자 영역을 CNN 으로 특징 추출 → BiLSTM 으로 왼쪽→오른쪽 순서를 읽는다.
     - CTC(Connectionist Temporal Classification) 디코딩으로 "ㅎㅎ안안녕--" 같은 출력을
       중복/공백을 정리해 "안녕" 으로 만든다. (글자 위치를 일일이 라벨링하지 않아도 학습 가능)
     - 언어별로 인식 모델이 따로 있다. ['ko', 'en'] 이면 한국어+영어 모델을 쓴다.

  YOLOE 와 비교하면: ① 검출 = "어디에", ② 인식 = "무엇" 에 해당한다.
  탐지 + 분류를 한 번에 하는 YOLO 와 달리, OCR 은 보통 두 모델을 이어 붙인다.

실행 예:
    py ocr_practice.py --mode basic                 # 기본: 글자 읽고 박스 그리기
    py ocr_practice.py --mode pipeline              # 검출 → 인식 두 단계를 따로 실행해보기
    py ocr_practice.py --mode robust                # 흐림/저해상도/회전 등에 얼마나 강한가
    py ocr_practice.py --mode find --find 유통기한    # 특정 단어 찾아 표시하기
    py ocr_practice.py --mode all                   # 위 전부

    내 이미지:  --source 내사진.jpg

처음 실행 시 EasyOCR 모델(검출 + 한국어/영어 인식, 약 100MB)이
C:\\Users\\<사용자>\\.EasyOCR 에 자동 다운로드된다.
결과 이미지는 이 폴더의 runs/ 에 저장된다.
"""

import argparse
import os
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# PyTorch 내부 경고(양자화 deprecated, pin_memory 등)는 실습과 무관하므로 숨긴다
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent

# 사용자가 입력한 경로는 '실행한 위치' 기준이므로, 폴더를 옮기기 전 위치를 기억해 둔다.
START_DIR = Path.cwd()


def user_path(p):
    """명령줄로 받은 상대경로를 실행 위치 기준 절대경로로 바꾼다. (URL 은 그대로)"""
    if p.startswith("http"):
        return p
    path = (START_DIR / p).resolve()
    if not path.exists():
        raise SystemExit(f"파일을 찾을 수 없습니다: {path}")
    return str(path)


os.chdir(HERE)
SAVE_DIR = HERE / "runs"

# 한글을 그림에 쓰기 위한 폰트 (Windows 기본 '맑은 고딕')
#   OpenCV 의 cv2.putText 는 한글을 못 그리므로 PIL 로 글자를 그린다.
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# EasyOCR 공식 저장소의 한국어 예제 이미지 (거리 간판 사진)
SIGN_URL = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/korean.png"


# ---------------------------------------------------------------------------
# 예제 이미지 준비
# ---------------------------------------------------------------------------
def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def make_label_image(path="label.jpg"):
    """
    실습용 '제품 라벨' 이미지를 직접 만든다.
    정답 텍스트를 우리가 알고 있으므로 OCR 결과가 맞았는지 바로 비교할 수 있다.
    """
    img = Image.new("RGB", (900, 560), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 880, 540], outline="black", width=4)
    d.text((50, 45), "PHYSICAL AI LAB", font=font(56), fill="black")
    d.text((50, 135), "제품명: 로봇 그리퍼 부품 A-17", font=font(40), fill="black")
    d.text((50, 205), "유통기한 2026.12.31", font=font(40), fill=(200, 0, 0))
    d.text((50, 275), "송장번호 1234-5678-9012", font=font(40), fill="black")
    d.text((50, 345), "주의: 뜨거움 HOT", font=font(40), fill=(200, 0, 0))
    d.text((50, 430), "작은 글씨는 인식이 어려울 수 있습니다 (size 22)", font=font(22), fill=(90, 90, 90))
    d.text((50, 475), "Small text: OCR accuracy drops (size 22)", font=font(22), fill=(90, 90, 90))
    img.save(path, quality=95)
    return path


def prepare_default_images():
    """기본 이미지 2장: 직접 만든 라벨 + 실제 간판 사진"""
    paths = []
    label = Path("label.jpg")
    if not label.exists():
        make_label_image(str(label))
    paths.append(str(label))

    sign = Path("korean_sign.png")
    if not sign.exists():
        print(f"  다운로드: {SIGN_URL}")
        sign.write_bytes(requests.get(SIGN_URL, timeout=30).content)
    paths.append(str(sign))
    return paths


# ---------------------------------------------------------------------------
# 시각화 / 출력 도우미
# ---------------------------------------------------------------------------
def draw_results(img_bgr, results, highlight=None):
    """
    OCR 결과를 이미지 위에 그린다.
      - 초록 박스: 인식된 글자 영역 (신뢰도가 낮을수록 빨간색에 가깝게)
      - 박스 위: 인식된 텍스트와 신뢰도
      - highlight 에 해당하는 결과는 굵은 파란 박스
    """
    img = img_bgr.copy()
    for i, (box, text, conf) in enumerate(results):
        pts = np.array(box, dtype=np.int32)
        # 신뢰도 1 → 초록, 0 → 빨강 (BGR)
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        thick = 2
        if highlight is not None and i in highlight:
            color, thick = (255, 80, 0), 5
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick)

    # 한글 라벨은 PIL 로 그린다
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font(max(14, img.shape[0] // 40))
    for box, text, conf in results:
        x, y = int(box[0][0]), int(box[0][1])
        label = f"{text} ({conf:.2f})"
        tw = d.textlength(label, font=f)
        ty = max(0, y - f.size - 6)
        d.rectangle([x, ty, x + tw + 6, ty + f.size + 6], fill=(0, 0, 0))
        d.text((x + 3, ty + 1), label, font=f, fill=(255, 255, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save(img, name):
    SAVE_DIR.mkdir(exist_ok=True)
    path = SAVE_DIR / name
    # cv2.imwrite 는 한글 경로를 못 쓰므로 imencode 후 직접 저장한다
    ok, buf = cv2.imencode(Path(name).suffix, img)
    path.write_bytes(buf.tobytes())
    print(f"  저장: {path}")


def imread(path):
    """한글 경로도 읽을 수 있는 cv2.imread 대체 함수"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def print_results(results, min_conf):
    """
    EasyOCR 결과 = [(box, text, confidence), ...]
      box  : 네 꼭짓점 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (왼쪽 위부터 시계방향)
             → 기울어진 글자도 표현할 수 있도록 사각형이 아닌 '사각 다각형'이다.
      text : 인식된 문자열
      conf : 인식 신뢰도 (0~1)
    """
    kept = [r for r in results if r[2] >= min_conf]
    print(f"  검출 {len(results)}개 / 신뢰도 {min_conf} 이상 {len(kept)}개")
    for box, text, conf in results:
        mark = "" if conf >= min_conf else "   ← 신뢰도 낮음(제외)"
        x, y = int(box[0][0]), int(box[0][1])
        print(f"   {conf:.2f}  ({x:4d},{y:4d})  {text}{mark}")
    return kept


def fix_digits(text):
    """
    OCR 후처리 예시: 숫자가 와야 할 자리에서 자주 헷갈리는 글자를 숫자로 바꾼다.
      O→0, o→0, I→1, l→1, S→5, B→8
    실제 결과에서도 '205Km' 를 '2O5Km' 로 읽는 일이 생긴다.
    숫자 바로 옆에 붙은 글자만 바꾸므로 일반 단어는 건드리지 않는다.
    ※ '2026.12.31' 을 '2026.12.37' 로 읽는 것처럼 숫자끼리 틀린 건 이런 규칙으로 못 고친다.
      날짜 형식 검사(월 1~12, 일 1~31) 같은 '도메인 지식' 검증이 추가로 필요하다.
    """
    table = str.maketrans("OoIlSB", "001158")
    return re.sub(r"(?<=\d)[OoIlSB]|[OoIlSB](?=\d)", lambda m: m.group().translate(table), text)


def reading_order(results, line_tol=20):
    """
    결과를 '사람이 읽는 순서'(위→아래, 왼→오른쪽)로 정렬한다.
    y 좌표가 line_tol 픽셀 이내면 같은 줄로 본다.
    """
    items = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
    lines, cur, cur_y = [], [], None
    for r in items:
        y = r[0][0][1]
        if cur_y is None or abs(y - cur_y) <= line_tol:
            cur.append(r)
            cur_y = y if cur_y is None else cur_y
        else:
            lines.append(sorted(cur, key=lambda r: r[0][0][0]))
            cur, cur_y = [r], y
    if cur:
        lines.append(sorted(cur, key=lambda r: r[0][0][0]))
    return [" ".join(r[1] for r in line) for line in lines]


# ---------------------------------------------------------------------------
# 1) 기본
# ---------------------------------------------------------------------------
def mode_basic(reader, sources, min_conf):
    print("\n[basic] 이미지에서 글자 읽기")
    for src in sources:
        print(f"\n  ▶ {Path(src).name}")
        t = time.perf_counter()
        # readtext(): 검출 + 인식을 한 번에 수행
        #   detail=1 (기본): (박스, 텍스트, 신뢰도) 반환 / detail=0: 텍스트만 반환
        results = reader.readtext(imread(src))
        print(f"  처리 시간: {time.perf_counter() - t:.2f}s")
        kept = print_results(results, min_conf)

        print("  [읽는 순서로 정리한 전체 텍스트]")
        for line in reading_order(kept):
            print(f"   | {line}")

        save(draw_results(imread(src), results), f"basic_{Path(src).stem}.jpg")

    # 참고: paragraph=True 를 주면 가까운 글자 박스들을 문단 단위로 합쳐준다 (신뢰도는 반환 안 함)
    #   reader.readtext(src, paragraph=True)


# ---------------------------------------------------------------------------
# 2) 두 단계를 따로 보기
# ---------------------------------------------------------------------------
def mode_pipeline(reader, src):
    print(f"\n[pipeline] 검출 → 인식 두 단계를 나눠서 실행: {Path(src).name}")
    img = imread(src)

    # ① 검출만: 글자가 '어디' 있는지
    #   horizontal_list: 수평 박스 [x_min, x_max, y_min, y_max]
    #   free_list      : 기울어진 박스 (네 꼭짓점)
    t = time.perf_counter()
    horizontal_list, free_list = reader.detect(img)
    t_det = time.perf_counter() - t
    h_boxes, f_boxes = horizontal_list[0], free_list[0]   # 이미지 1장이므로 [0]
    print(f"  ① 검출 (CRAFT): {t_det:.2f}s → 수평 박스 {len(h_boxes)}개, 기울어진 박스 {len(f_boxes)}개")

    vis = img.copy()
    for x1, x2, y1, y2 in h_boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for pts in f_boxes:
        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], True, (255, 0, 0), 2)
    save(vis, "pipeline_1_detect.jpg")
    print("     → 이 단계 결과 이미지에는 박스만 있고 글자 내용은 없다.")

    # ② 인식만: 검출된 박스 안의 글자가 '무엇'인지
    t = time.perf_counter()
    results = reader.recognize(img, horizontal_list=h_boxes, free_list=f_boxes)
    t_rec = time.perf_counter() - t
    print(f"  ② 인식 (CRNN): {t_rec:.2f}s → {len(results)}개 텍스트")
    for _, text, conf in results:
        print(f"     {conf:.2f}  {text}")
    save(draw_results(img, results), "pipeline_2_recognize.jpg")

    # ① 의 박스 일부만 골라 인식시켜보면, 검출이 놓친 글자는 인식도 절대 못 한다는 걸 알 수 있다.
    #   → OCR 정확도는 '검출 품질'에 크게 좌우된다.


# ---------------------------------------------------------------------------
# 3) 이미지 품질에 따른 성능 변화
# ---------------------------------------------------------------------------
def mode_robust(reader, src):
    print(f"\n[robust] 이미지를 망가뜨리면 OCR 이 얼마나 버틸까: {Path(src).name}")
    img = imread(src)
    h, w = img.shape[:2]

    def rotate(im, deg):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(im, m, (w, h), borderValue=(255, 255, 255))

    def low_res(im, scale):
        small = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def low_contrast(im):
        return cv2.addWeighted(im, 0.25, np.full_like(im, 160), 0.75, 0)

    def noise(im):
        rng = np.random.default_rng(0)
        return np.clip(im.astype(np.int16) + rng.normal(0, 40, im.shape), 0, 255).astype(np.uint8)

    variants = {
        "original": img,
        "blur": cv2.GaussianBlur(img, (9, 9), 0),        # 초점이 안 맞은 사진
        "low_res_30%": low_res(img, 0.3),                # 멀리서 찍은 사진
        "rotate_15deg": rotate(img, 15),                 # 기울어진 사진
        "rotate_90deg": rotate(img, 90),                 # 옆으로 누운 사진
        "low_contrast": low_contrast(img),               # 흐릿한 인쇄/역광
        "noise": noise(img),                             # 어두운 곳 노이즈
    }

    print(f"\n  {'변형':<14} | 검출 | 평균 신뢰도 | 인식된 텍스트 (앞부분)")
    print("  " + "-" * 80)
    tiles = []
    for name, im in variants.items():
        results = reader.readtext(im)
        avg = np.mean([r[2] for r in results]) if results else 0.0
        joined = " / ".join(r[1] for r in results)
        print(f"  {name:<14} | {len(results):4d} | {avg:10.2f} | {joined[:60]}")
        tile = draw_results(im, results)
        cv2.putText(tile, name, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(tile, name, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        tiles.append(cv2.resize(tile, (w // 2, h // 2)))

    # 결과를 격자로 모아서 한 장으로 저장
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    save(grid, "robust_grid.jpg")

    # 해결책: 회전 문제는 rotation_info 로 완화할 수 있다.
    #   EasyOCR 이 90/180/270도 돌린 버전도 시도해 보고 가장 신뢰도 높은 결과를 고른다 (대신 느려진다).
    results = reader.readtext(variants["rotate_90deg"], rotation_info=[90, 180, 270])
    avg = np.mean([r[2] for r in results]) if results else 0.0
    print(f"  {'rotate_90 + rotation_info':<14} | {len(results):4d} | {avg:10.2f} | "
          f"{' / '.join(r[1] for r in results)[:60]}")
    print("\n  → 어떤 변형에서 가장 크게 무너지는지 보고, 실제 카메라 세팅에서 무엇을 신경 써야 할지 생각해보자.")
    print("  → 주의: rotate_15deg 처럼 '평균 신뢰도는 높은데 글자가 조각조각' 나는 경우가 있다.")
    print("    신뢰도는 '이 조각을 이 글자로 읽은 확신'일 뿐, 문장 전체가 맞았다는 뜻이 아니다.")


# ---------------------------------------------------------------------------
# 4) 특정 단어 찾기
# ---------------------------------------------------------------------------
def mode_find(reader, sources, keyword, min_conf):
    print(f"\n[find] '{keyword}' 가 들어간 글자 찾기")
    # 로봇 응용 예: "유통기한" 옆의 날짜 읽기, 택배 상자에서 "송장번호" 찾기, 버튼 라벨 찾기
    for src in sources:
        results = [r for r in reader.readtext(imread(src)) if r[2] >= min_conf]
        # 공백을 무시하고 비교 (OCR 은 띄어쓰기를 자주 틀린다)
        key = keyword.replace(" ", "").lower()
        hits = [i for i, r in enumerate(results) if key in r[1].replace(" ", "").lower()]
        print(f"  ▶ {Path(src).name}: {len(hits)}개 발견")
        for i in hits:
            box, text, conf = results[i]
            cx = int(np.mean([p[0] for p in box]))
            cy = int(np.mean([p[1] for p in box]))
            print(f"     '{text}' (신뢰도 {conf:.2f}) 중심 좌표 ({cx}, {cy})")
            # 키워드와 값이 한 박스로 묶여 나오는 경우가 많다 → 키워드 뒤 부분을 '값'으로 본다
            value = re.split(re.escape(keyword), text, maxsplit=1, flags=re.I)[-1].strip(" :")
            if value and value != text:
                print(f"       값 후보: '{value}'  → 후처리: '{fix_digits(value)}'")
            # 같은 줄에서 오른쪽에 있는 텍스트 = 그 항목의 '값'일 가능성이 높다
            same_line = [r for r in results if r is not results[i]
                         and abs(r[0][0][1] - box[0][1]) < 20 and r[0][0][0] > box[0][0]]
            if same_line:
                print(f"       같은 줄 오른쪽: {[r[1] for r in same_line]}")
        if hits:
            save(draw_results(imread(src), results, highlight=set(hits)), f"find_{Path(src).stem}.jpg")


def main():
    parser = argparse.ArgumentParser(description="EasyOCR 실습")
    parser.add_argument("--mode", default="all", choices=["basic", "pipeline", "robust", "find", "all"])
    parser.add_argument("--source", nargs="+", default=None, help="이미지 경로들 (생략 시 예제 이미지)")
    parser.add_argument("--langs", nargs="+", default=["ko", "en"], help="인식 언어 (예: ko en / en / ja en)")
    parser.add_argument("--find", default="유통기한", help="find 모드에서 찾을 단어")
    parser.add_argument("--min-conf", type=float, default=0.3, help="이 신뢰도 미만 결과는 제외")
    parser.add_argument("--gpu", action="store_true", help="GPU 사용 (CUDA 가 있을 때)")
    args = parser.parse_args()

    sources = [user_path(p) for p in args.source] if args.source else prepare_default_images()

    # Reader 생성 = 모델 로드 (처음 한 번만, 시간이 걸린다)
    #   lang_list: 인식할 언어. 'ko' 와 'en' 은 함께 쓸 수 있지만, 모든 조합이 되는 건 아니다.
    #              (예: 'ko' 와 'ja' 는 서로 다른 인식 모델이라 동시에 못 쓴다)
    #   gpu=False: CPU 로 실행. GPU 가 있으면 --gpu 로 훨씬 빨라진다.
    import easyocr
    print(f"EasyOCR Reader 로드 (언어: {args.langs})")
    reader = easyocr.Reader(args.langs, gpu=args.gpu, verbose=False)

    if args.mode in ("basic", "all"):
        mode_basic(reader, sources, args.min_conf)
    if args.mode in ("pipeline", "all"):
        mode_pipeline(reader, sources[0])
    if args.mode in ("robust", "all"):
        mode_robust(reader, sources[0])
    if args.mode in ("find", "all"):
        mode_find(reader, sources, args.find, args.min_conf)

    # -----------------------------------------------------------------------
    # 실습 과제 아이디어
    #   1. basic 결과에서 작은 글씨(size 22) 줄의 정확도를 큰 글씨와 비교해보자.
    #   2. --langs en 으로만 돌리면 한국어는 어떻게 나올까?
    #   3. robust 에서 가장 치명적인 변형은 무엇인가? 카메라를 설치한다면 무엇을 조심해야 할까?
    #   4. 휴대폰으로 과자 봉지, 영수증, 책 표지를 찍어 --source 로 넣어보자.
    #   5. find 모드로 '송장번호' 를 찾고, 같은 줄 오른쪽의 번호만 뽑아내 보자.
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
