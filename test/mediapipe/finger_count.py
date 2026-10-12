"""
MediaPipe 실습 — 손가락으로 숫자 세기
=====================================

웹캠에 손가락을 펴서 보여주면 몇 개를 폈는지 세어 화면에 크게 띄운다. (양손이면 0~10)

전체 흐름:
  웹캠 프레임 ──▶ [MediaPipe Hand Landmarker] ──▶ 손마다 관절 21개 좌표 ──▶ [우리가 짠 규칙] ──▶ 숫자
                  (딥러닝: 손 찾기 + 관절 위치)                                 (각도 계산)

  - MediaPipe 는 "관절이 어디 있는지"까지만 알려준다.
  - "손가락이 펴졌나?"를 판단하는 건 그 좌표로 우리가 직접 계산한다. ← 이 실습의 핵심

MediaPipe Hand Landmarker 내부는 두 단계다:
  ① Palm Detector  : 이미지 전체에서 '손바닥'을 찾는다 (손가락보다 모양 변화가 적어서 찾기 쉽다)
  ② Hand Landmark  : 찾은 손 영역을 잘라서 관절 21개의 3D 좌표를 회귀한다
  영상 모드에서는 이전 프레임의 손 위치로 ② 만 돌리고 ① 은 손을 놓쳤을 때만 다시 돌려서 빠르다.

손 관절 21개 번호 (외워두면 편하다):

           8   12  16  20        ← TIP  (손가락 끝)
           |   |   |   |
           7   11  15  19        ← DIP
     4     |   |   |   |
     |     6   10  14  18        ← PIP  (손가락 가운데 마디)
     3     |   |   |   |
      \\    5---9---13--17       ← MCP  (손가락 뿌리)
       2    \\            /
        \\    \\          /
         1-----\\--------/
                0                ← WRIST (손목)

    엄지: 1(CMC) 2(MCP) 3(IP) 4(TIP)
    검지: 5 6 7 8 / 중지: 9 10 11 12 / 약지: 13 14 15 16 / 새끼: 17 18 19 20

실행:
    py finger_count.py                 # 웹캠 (기본 0번)
    py finger_count.py --camera 1      # 다른 카메라
    py finger_count.py --demo          # 웹캠 없이 예제 손 사진으로 테스트
    py finger_count.py --image a.jpg   # 내 사진으로 테스트

웹캠 창 조작:  q / ESC = 종료,  s = 화면 저장,  d = 디버그(관절 번호·각도) 표시 켜기/끄기
"""

import argparse
import math
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
START_DIR = Path.cwd()  # 사용자가 입력한 경로는 '실행한 위치' 기준
os.chdir(HERE)
SAVE_DIR = HERE / "runs"

# MediaPipe 1.x 는 모델 파일(.task)을 직접 받아서 경로로 넘겨줘야 한다
MODEL_PATH = HERE / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# --demo 에서 쓰는 예제 손 사진 (MediaPipe 공식 예제)
DEMO_IMAGES = {
    "pointing_up.jpg": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg",
    "victory.jpg": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/victory.jpg",
    "thumbs_up.jpg": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg",
    "woman_hands.jpg": "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
}

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 한글 표시용 (cv2.putText 는 한글을 못 그린다)

# 손가락별 관절 번호: (뿌리, 가운데, 끝쪽 마디, 끝)
FINGERS = {
    "엄지": (1, 2, 3, 4),
    "검지": (5, 6, 7, 8),
    "중지": (9, 10, 11, 12),
    "약지": (13, 14, 15, 16),
    "새끼": (17, 18, 19, 20),
}

# '펴졌다'고 판단할 기준 (finger_states 설명 참고)
EXTEND_RATIO = 1.1     # 검지~새끼: |손목→끝| / |손목→가운데 마디| 가 이보다 크면 펴진 것
THUMB_ANGLE = 150      # 엄지: IP 관절 각도(도)가 이보다 크면 펴진 것. 180 = 완전히 일자

# 최근 몇 프레임의 결과로 다수결을 할지 (값이 깜빡이는 것을 막는 안정화)
SMOOTH_FRAMES = 7


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------
def download(path, url):
    if not Path(path).exists():
        print(f"  다운로드: {url}")
        Path(path).write_bytes(requests.get(url, timeout=60).content)
    return str(path)


def create_landmarker(running_mode):
    """
    HandLandmarker 생성.
      running_mode:
        IMAGE : 사진 한 장씩 독립적으로 처리
        VIDEO : 연속 프레임. 이전 프레임 정보를 활용해 추적하므로 빠르고 안정적
                (대신 detect_for_video 에 '계속 증가하는' 타임스탬프를 넘겨야 한다)
    """
    download(MODEL_PATH, MODEL_URL)
    options = vision.HandLandmarkerOptions(
        # model_asset_path 로 경로를 넘기면 MediaPipe 내부(C++)가 한글 경로를 못 연다.
        # 그래서 파이썬에서 파일을 읽어 내용(bytes)을 직접 넘긴다.
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=running_mode,
        num_hands=2,                          # 최대 몇 손까지 찾을지
        min_hand_detection_confidence=0.5,    # ① 손바닥 검출 신뢰도 기준
        min_hand_presence_confidence=0.5,     # ② 랜드마크 단계에서 '손이 여전히 있다'고 볼 기준
        min_tracking_confidence=0.5,          # VIDEO 모드에서 추적을 이어갈 기준
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# 핵심: 손가락이 펴졌는지 판단하기
# ---------------------------------------------------------------------------
def angle_at(a, b, c):
    """
    b 를 꼭짓점으로 하는 각 a-b-c (도 단위).
    벡터 BA, BC 사이 각도 = acos( BA·BC / |BA||BC| )
      180도 → a, b, c 가 일직선 (손가락이 쭉 펴짐)
       90도 → 직각으로 굽음
    """
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def dist(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def finger_states(lm):
    """
    손 하나의 관절 21개(lm) → {손가락 이름: (펴짐 여부, 판단에 쓴 값)}

    가장 단순한 방법은 "끝(TIP)의 y 가 가운데 마디(PIP)보다 위에 있으면 펴짐" 인데,
    손을 옆으로 눕히거나 아래로 향하면 바로 틀린다. 그래서 방향과 무관한 '3D 거리/각도'를 쓴다.

    [검지~새끼] 거리 비율 = |손목→끝| / |손목→가운데 마디|
      - 펴진 손가락: 끝이 손목에서 멀리 뻗어 있다 → 비율이 크다 (예제 사진에서 1.19 이상)
      - 접힌 손가락: 끝이 손바닥 쪽으로 말려 들어온다 → 비율이 작다 (0.99 이하)
      ※ 처음에는 '관절 각도'로 판단했는데, 힘을 빼고 편 손가락은 각도가 145~155도로
        생각보다 많이 굽어 있어서 접힌 손가락(최대 146도)과 구분이 잘 안 됐다.
        거리 비율은 두 그룹이 확실히 갈라져서 더 안정적이다. (--debug 로 직접 확인해보자)

    [엄지] 다른 손가락과 구조가 달라서 두 조건을 함께 본다
      1) IP 관절(3)의 각도가 일자에 가깝고
      2) 엄지 끝(4)이 IP(3)보다 검지 뿌리(5)에서 더 멀다 (손바닥 위로 접혀 있지 않다)

    여기서 lm 은 hand_world_landmarks (미터 단위 실제 3D 좌표, 손 중심이 원점)를 쓴다.
      - hand_landmarks      : 이미지 기준 정규화 좌표 (x, y 는 0~1). 화면에 그릴 때 사용.
      - hand_world_landmarks: 실제 크기 3D 좌표. 원근·화면 비율 왜곡이 없어 거리·각도 계산에 적합.
    """
    states = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        if name == "엄지":
            ang = angle_at(lm[pip], lm[dip], lm[tip])            # 2-3-4 각도
            away = dist(lm[tip], lm[5]) > dist(lm[dip], lm[5])   # 끝이 IP 보다 검지 뿌리에서 더 멀다
            states[name] = (ang > THUMB_ANGLE and away, ang)
        else:
            ratio = dist(lm[0], lm[tip]) / (dist(lm[0], lm[pip]) + 1e-9)
            states[name] = (ratio > EXTEND_RATIO, ratio)
    return states


def analyze(result):
    """HandLandmarkerResult → 손별 정보 리스트"""
    hands = []
    for i, world in enumerate(result.hand_world_landmarks):
        # handedness: 왼손/오른손 판정. MediaPipe 는 '좌우 반전된(셀카) 이미지' 를 가정한다.
        #   그래서 웹캠 프레임을 cv2.flip 으로 거울처럼 뒤집어서 넣으면 실제 손과 일치한다.
        side = result.handedness[i][0].category_name   # "Left" / "Right"
        states = finger_states(world)
        hands.append(dict(
            side="왼손" if side == "Left" else "오른손",
            states=states,
            count=sum(up for up, _ in states.values()),
            image_lm=result.hand_landmarks[i],
        ))
    return hands


# ---------------------------------------------------------------------------
# 화면 그리기
# ---------------------------------------------------------------------------
def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_hand(img, hand, debug=False):
    """관절과 뼈대 그리기. 펴진 손가락은 초록, 접힌 손가락은 회색."""
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand["image_lm"]]

    finger_of = {}
    for name, idx in FINGERS.items():
        for j in idx:
            finger_of[j] = name
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        s, e = conn.start, conn.end
        name = finger_of.get(e)
        up = name is not None and hand["states"][name][0]
        color = (0, 220, 0) if up else (160, 160, 160)
        cv2.line(img, pts[s], pts[e], color, 3)
    for j, p in enumerate(pts):
        cv2.circle(img, p, 5, (255, 255, 255), -1)
        cv2.circle(img, p, 5, (0, 0, 0), 1)
        if debug:
            cv2.putText(img, str(j), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if debug:
        # 판단에 쓴 값 표시: 엄지는 각도(°), 나머지는 거리 비율
        for name, (mcp, pip, dip, tip) in FINGERS.items():
            x, y = pts[tip]
            cv2.putText(img, fmt_value(name, hand["states"][name][1]), (x - 20, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    return pts


def fmt_value(name, v):
    return f"{v:.0f}deg" if name == "엄지" else f"{v:.2f}"


def draw_overlay(img, hands, total, fps=None):
    """큰 숫자 + 손별 손가락 상태를 한글로 표시"""
    for hand in hands:
        hand["pts"] = draw_hand(img, hand, debug=DEBUG)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    H, W = img.shape[:2]
    S = min(H, W)   # 세로/가로 사진 모두에서 적당한 크기가 되도록 짧은 변 기준

    # 왼쪽 위: 큰 숫자
    box = int(S * 0.3)
    big = font(int(box * 0.8))
    text = "-" if total is None else str(total)
    d.rounded_rectangle([15, 15, 15 + box, 15 + box], radius=20, fill=(0, 0, 0))
    l, t, r, b = d.textbbox((0, 0), text, font=big)
    d.text((15 + (box - (r - l)) / 2 - l, 15 + (box - (b - t)) / 2 - t), text, font=big, fill=(255, 230, 0))

    # 손마다: 손목 아래에 "오른손 3" + 손가락 상태 (화면 밖으로 나가지 않게 위치 보정)
    small = font(max(16, S // 30))
    for hand in hands:
        fingers = " ".join(f"{n}{'●' if up else '○'}" for n, (up, _) in hand["states"].items())
        label = f"{hand['side']} {hand['count']}"
        bw = d.textlength(fingers, font=small) + 16
        bh = small.size * 2 + 16
        x, y = hand["pts"][0]
        x = int(min(max(x - bw / 2, 5), W - bw - 5))
        y = int(min(y + 15, H - bh - 5))
        d.rectangle([x, y, x + bw, y + bh], fill=(0, 0, 0))
        d.text((x + 8, y + 4), label, font=small, fill=(255, 230, 0))
        d.text((x + 8, y + 8 + small.size), fingers, font=small, fill=(255, 255, 255))

    if not hands:
        d.text((20, 25 + box), "손을 보여주세요", font=small, fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
    if fps is not None:
        d.text((img.shape[1] - 120, 10), f"FPS {fps:.0f}", font=small, fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save(img, name):
    SAVE_DIR.mkdir(exist_ok=True)
    path = SAVE_DIR / name
    ok, buf = cv2.imencode(".jpg", img)   # 한글 경로 대응 (cv2.imwrite 는 한글 경로 실패)
    path.write_bytes(buf.tobytes())
    print(f"  저장: {path}")


def to_mp_image(bgr):
    """OpenCV(BGR) → MediaPipe Image(RGB)"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ---------------------------------------------------------------------------
# 실행 모드
# ---------------------------------------------------------------------------
def run_images(paths):
    """사진으로 테스트 (웹캠이 없을 때 로직 확인용)"""
    landmarker = create_landmarker(vision.RunningMode.IMAGE)
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = landmarker.detect(to_mp_image(img))
        hands = analyze(result)
        total = sum(h["count"] for h in hands) if hands else None
        print(f"\n  ▶ {Path(p).name}: 손 {len(hands)}개 → 숫자 {total}")
        for h in hands:
            detail = ", ".join(f"{n}:{'펴짐' if up else '접힘'}({fmt_value(n, v)})" for n, (up, v) in h["states"].items())
            print(f"     {h['side']} {h['count']}개  [{detail}]")
        # 사진은 셀카처럼 반전되지 않았으므로 왼손/오른손 표시가 반대로 나올 수 있다
        save(draw_overlay(img, hands, total), f"result_{Path(p).stem}.jpg")


def run_webcam(camera):
    global DEBUG
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)  # Windows 에서는 DSHOW 백엔드가 빨리 열린다
    if not cap.isOpened():
        raise SystemExit(f"카메라 {camera} 번을 열 수 없습니다. --camera 1 처럼 다른 번호를 시도해보세요.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = create_landmarker(vision.RunningMode.VIDEO)
    history = deque(maxlen=SMOOTH_FRAMES)
    t0, prev, fps = time.perf_counter(), time.perf_counter(), 0.0
    print("웹캠 실행 중 — q/ESC: 종료, s: 저장, d: 디버그 표시")

    # 단계별 소요 시간 기록 (종료 시 요약 출력). 모델 추론만이 아니라 '실제 카메라 루프' 전체를 잰다.
    timing = {"프레임 받기": [], "모델 추론": [], "판단+그리기+표시": [], "한 바퀴 전체": []}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Finger Count", cv2.WINDOW_NORMAL)
    while True:
        t_loop = time.perf_counter()
        ok, frame = cap.read()       # 카메라가 다음 프레임을 줄 때까지 기다리는 시간도 포함된다
        if not ok:
            break
        t_cap = time.perf_counter()
        frame = cv2.flip(frame, 1)   # 거울 모드: 보기 자연스럽고, MediaPipe 의 왼손/오른손 판정과도 맞다

        # VIDEO 모드는 ms 단위 타임스탬프가 계속 증가해야 한다
        ts = int((time.perf_counter() - t0) * 1000)
        result = landmarker.detect_for_video(to_mp_image(frame), ts)
        t_inf = time.perf_counter()
        hands = analyze(result)

        # 안정화: 최근 N 프레임 중 가장 많이 나온 숫자를 표시 (손가락을 움직이는 중간 값이 깜빡이지 않게)
        history.append(sum(h["count"] for h in hands) if hands else None)
        total = Counter(history).most_common(1)[0][0]

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now

        view = draw_overlay(frame, hands, total, fps)
        cv2.imshow("Finger Count", view)
        key = cv2.waitKey(1) & 0xFF
        t_end = time.perf_counter()
        for k, v in zip(timing, (t_cap - t_loop, t_inf - t_cap, t_end - t_inf, t_end - t_loop)):
            timing[k].append(v * 1000)
        if key in (ord("q"), 27) or cv2.getWindowProperty("Finger Count", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            save(view, f"webcam_{int(time.time())}.jpg")
        if key == ord("d"):
            DEBUG = not DEBUG

    cap.release()
    cv2.destroyAllWindows()
    print_timing(timing, w, h)


def print_timing(timing, w, h):
    """
    실제 웹캠 루프의 단계별 시간 요약.
    정지 이미지로 잰 '모델 추론 시간'과 달리, 카메라 대기·손을 다시 찾는 시간·화면 표시까지 모두 들어간다.
    처음 30프레임은 카메라 초기화로 느리므로 뺀다.
    """
    n = len(timing["한 바퀴 전체"])
    if n <= 30:
        print("측정하기에 프레임이 너무 적습니다 (30프레임 이상 실행해 주세요).")
        return
    print(f"\n[실측] 웹캠 {w}x{h}, {n - 30}프레임 기준 (중앙값 / 상위 5%)")
    for k, v in timing.items():
        v = sorted(v[30:])
        print(f"  {k:<14} {v[len(v) // 2]:6.1f}ms / {v[int(len(v) * 0.95)]:6.1f}ms")
    loop = sorted(timing["한 바퀴 전체"][30:])[(n - 30) // 2]
    print(f"  → 실제 FPS 약 {1000 / loop:.1f}")
    print("  ※ '프레임 받기'가 크면 카메라 자체 속도(보통 30fps = 33ms)에 묶여 있는 것이다.")


DEBUG = False


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="MediaPipe 손가락 숫자 세기")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호")
    parser.add_argument("--image", nargs="+", default=None, help="웹캠 대신 사진으로 테스트")
    parser.add_argument("--demo", action="store_true", help="예제 손 사진으로 테스트")
    parser.add_argument("--debug", action="store_true", help="관절 번호와 각도 표시")
    args = parser.parse_args()
    DEBUG = args.debug

    if args.demo:
        run_images([download(name, url) for name, url in DEMO_IMAGES.items()])
    elif args.image:
        paths = []
        for p in args.image:
            path = (START_DIR / p).resolve()
            if not path.exists():
                raise SystemExit(f"파일을 찾을 수 없습니다: {path}")
            paths.append(str(path))
        run_images(paths)
    else:
        run_webcam(args.camera)

    # -----------------------------------------------------------------------
    # 실습 과제 아이디어
    #   1. --debug 로 값을 보면서 EXTEND_RATIO 를 0.9 / 1.4 로 바꿔보자. 무엇이 달라지나?
    #   2. finger_states 를 "TIP 의 y < PIP 의 y" 방식으로 바꾸고 손을 옆으로 눕혀보자. 왜 틀릴까?
    #   3. SMOOTH_FRAMES 를 1 로 바꾸면 숫자가 얼마나 깜빡이나? 15 로 하면 반응이 얼마나 느려지나?
    #   4. 엄지+새끼만 펴면 '6'(한국식 숫자 표현)으로 인식하도록 규칙을 추가해보자.
    #   5. 숫자에 따라 다른 동작을 하게 만들어보자 (예: 1 → 사진 저장, 5 → 종료).
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
