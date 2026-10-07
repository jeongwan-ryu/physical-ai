"""실행 진입점: 셀프테스트 · 녹화 영상 데모 · 라즈베리 파이 실시간 실행

================================================================================
    python run_demo.py --selftest
        합성 영상으로 전체 파이프라인을 점검한다. 카메라·모델 없이 PC에서 바로 돌릴 수 있다.
        코드를 고친 뒤에는 항상 이것부터 돌려서 망가진 곳이 없는지 확인하자.

    python run_demo.py --video sofa.mp4 --led on --weights best.pt --out demo_out
    python run_demo.py --images frames/ --led off --out demo_out
        녹화 영상·사진에 전처리(+검출)를 적용하고, 원본과 전처리 결과를 위아래로 붙인
        비교 이미지를 저장한다. --led는 녹화 당시 LED를 켰는지.

    python run_demo.py --live --weights model.hef
        라즈베리 파이 5에서 카메라·LED·Hailo로 실시간 실행 (Ctrl+C로 종료).
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np

from offline_tools import CopyPaster, Cutout, PhysicalAugmenter
from preprocess import (CLASS_NAMES, CameraGeometry, Config, FrameMeta, LedController,
                        MultiFrameConfirmer, OnboardPipeline, Preprocessor, boxes_frame_to_roi,
                        boxes_roi_to_tile, imread, imwrite, merge_tiles, split_tiles, tile_offsets,
                        tiles_to_hailo, to_linear, to_srgb)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 검출기 고르기                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class UltralyticsDetector:
    """PC에서 녹화 영상으로 검증할 때 쓰는 YOLO (.pt / .onnx). Hailo NPU에서는 동작하지 않는다.

    결과를 OnboardPipeline이 원하는 형식 (타일별 (N,6) [x1,y1,x2,y2,conf,cls])으로 바꿔 준다.
    conf는 낮게(0.2) 거르고 클래스별 기준은 preprocess.filter_detections에서 적용한다.
    """

    def __init__(self, weights: str, imgsz=(480, 672), conf: float = 0.2, device=None):
        from ultralytics import YOLO          # 학습용 PC에만 설치하는 패키지라 여기서 import
        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def __call__(self, tiles):
        results = self.model.predict(list(tiles), imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            out.append(np.concatenate([
                b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None],
            ], axis=1).astype(np.float32))
        return out


def make_detector(weights: str, tile_hw=(480, 672)):
    """모델 파일 확장자로 검출기 선택: .hef → Hailo(Pi), 그 외(.pt/.onnx) → Ultralytics(PC)."""
    if weights.lower().endswith(".hef"):
        from rpi_hardware import HailoDetector
        return HailoDetector(weights)
    return UltralyticsDetector(weights, imgsz=tile_hw)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 셀프테스트                                                                   ║
# ║    가짜 장면과 가짜 검출기로 "각 단계가 기대한 대로 동작하는지"를 확인한다.      ║
# ║    모델 성능이 아니라 코드 배선(순서·좌표 변환·필터 규칙)을 검사하는 것이다.     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 합성 장면에 넣을 물체들의 위치 (ROI 좌표 x1, y1, x2, y2)
OBJ_ROI = (691, 314, 709, 326)       # 에어팟 (18×12px) — 확정되어야 함
GLARE_ROI = (632, 352, 648, 368)     # LED 반사점 — 에어팟으로 착각되지만 걸러져야 함
KIBBLE_ROI = (300, 300, 312, 310)    # 고양이 사료 (방해 물체) — 결과에서 빠져야 함


def synth_frame(rng, led_on: bool) -> np.ndarray:
    """소파 밑 유광 타일 바닥을 흉내 낸 합성 프레임 (1280×720).

    회색 타일 + 줄눈, 위쪽 240행은 어두운 소파 밑면, LED를 켜면 비네팅과 반사점,
    끄면 거의 깜깜(2 lux). 가운데쯤 흰 타원(에어팟)과 센서 노이즈를 넣는다.
    """
    H, W = 720, 1280
    img = np.full((H, W, 3), 175, np.uint8)
    for x in range(0, W, 160):
        cv2.line(img, (x, 240), (x, H), (110, 110, 110), 2)          # 세로 줄눈
    for y in range(300, H, 90):
        cv2.line(img, (0, y), (W, y), (110, 110, 110), 2)            # 가로 줄눈
    img[:240] = 60                                                    # 소파 밑면
    lin = to_linear(img)                                              # 빛 계산은 선형 공간에서
    if led_on:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - 640) / 640) ** 2 + ((yy - 720) / 500) ** 2
        lin *= np.clip(1.4 - 0.9 * r2, 0.2, 2.0)[..., None]            # 비네팅 (아래 가운데가 가장 밝음)
        gy, gx = np.mgrid[0:H, 0:W]
        lin += (3.0 * np.exp(-((gx - 640) ** 2 + (gy - 600) ** 2) / (2 * 16.0)))[..., None]  # 반사점
    else:
        lin *= 0.03                                                   # LED 없이 2 lux
    x1, y1, x2, y2 = OBJ_ROI
    cv2.ellipse(lin, ((x1 + x2) // 2, (y1 + y2) // 2 + 240), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360,
                (0.85, 0.85, 0.85) if led_on else (0.03, 0.03, 0.03), -1)
    lin += rng.normal(0, 0.01, lin.shape).astype(np.float32)          # 센서 노이즈
    return to_srgb(lin)


class FakeDetector:
    """합성 장면의 정답 위치를 그대로 돌려주는 가짜 검출기.

    - 에어팟: conf 0.6 → 기준 0.35를 넘으므로 통과해야 함
    - 반사점: conf 0.5로 '에어팟'이라고 착각 → 반사 마스크 안이라 기준이 0.55로 올라가 걸러져야 함
    - 사료:   conf 0.9 방해 물체 클래스 → 확신도와 상관없이 결과에서 빠져야 함
    LED가 꺼진 프레임(lit=False)은 너무 어두워서 아무것도 못 찾는다고 가정한다.
    """

    def __init__(self, cfg: Config):
        self.xs = tile_offsets(cfg.camera.width, cfg.pre.tile_w, cfg.pre.tile_overlap)
        self.tw = cfg.pre.tile_w
        self.lit = False            # 테스트 루프가 "이번 프레임은 LED 켠 상태"인지 알려 줌

    def __call__(self, tiles):
        out = []
        for tile, x0 in zip(tiles, self.xs):
            dets = []
            if self.lit:
                for (x1, y1, x2, y2), conf, cls in ((OBJ_ROI, 0.6, 0), (GLARE_ROI, 0.5, 0), (KIBBLE_ROI, 0.9, 6)):
                    if x1 >= x0 and x2 <= x0 + self.tw:                 # 이 타일 안에 온전히 있는 것만
                        dets.append([x1 - x0, y1, x2 - x0, y2, conf, cls])   # 타일 좌표로
            out.append(np.array(dets, np.float32).reshape(-1, 6))
        return out


def check(name, cond):
    print(("  ✓ " if cond else "  ✗ ") + name)
    if not cond:
        raise SystemExit(f"셀프테스트 실패: {name}")


def selftest(out_dir):
    cfg = Config()
    rng = np.random.default_rng(0)

    print("[1] 단위 점검")
    check("tile_offsets(1280, 672, 64) == [0, 608]", tile_offsets(1280, 672, 64) == [0, 608])
    v = np.arange(256, dtype=np.uint8)
    check("sRGB↔선형 왕복 오차 ≤ 1", int(np.abs(to_srgb(to_linear(v)).astype(int) - v).max()) <= 1)
    geo = CameraGeometry(cfg.camera)
    far, ok = geo.pixel_to_floor([[640, cfg.pre.roi_top]])
    check(f"ROI 맨 윗줄이 1m보다 먼 바닥 (지평선 {geo.horizon_row():.0f}행, ROI {cfg.pre.roi_top}행 → {far[0, 0]:.2f}m)",
          bool(ok[0] and far[0, 0] > 1.0))
    gain = np.random.default_rng(1).uniform(0.5, 2.5, (480, 1280, 1)).astype(np.float32)
    roi = rng.integers(0, 256, (480, 1280, 3), dtype=np.uint8)
    fast = Preprocessor(cfg.pre, gain).apply_flatfield(roi).astype(int)
    exact = to_srgb(to_linear(roi) * gain).astype(int)
    check("빠른 플랫필드(감마 영역 곱셈) = 선형 공간 계산 (오차 ≤ 1)", int(np.abs(fast - exact).max()) <= 1)
    tiles, _ = split_tiles(roi, cfg.pre)
    hx = tiles_to_hailo(tiles)
    check("Hailo 입력: (2,480,672,3) uint8 RGB", hx.shape == (2, 480, 672, 3) and hx.dtype == np.uint8
          and (hx[0, 0, 0] == roi[0, 0, ::-1]).all())
    floor, valid = geo.pixel_to_floor([[640, 700], [640, 100]])
    check("화면 아래 점은 바닥 앞쪽, 지평선 위 점은 무효", bool(valid[0] and floor[0, 0] > 0 and not valid[1]))

    led = LedController(cfg.led)
    cmds = [led.step(33, 8, i * 0.1) for i in range(5)]           # 노출 33ms × 게인 8 = 264 > 100 (어두움)
    check("어두우면 5프레임 뒤 LED 켜짐", cmds[-1].led_on and not cmds[-2].led_on)

    # 타일0의 오른쪽 끝(600~640)과 타일1의 왼쪽 끝(608+0~32)에서 잡힌 같은 물체 → 1개로 합쳐져야 함
    d = merge_tiles([np.array([[600, 10, 640, 30, 0.9, 0]], np.float32),
                     np.array([[0, 10, 32, 30, 0.7, 0]], np.float32)], [0, 608], 0.6)
    check("타일 경계 중복 박스 병합", len(d) == 1)

    mf = MultiFrameConfirmer(0.05, 3, 5)
    got = [mf.update(i, [(0, np.array([1.0, 0.0]), 0.6)]) for i in range(1, 5)]
    check("3프레임째 확정, 이후 중복 알림 없음", [len(g) for g in got] == [0, 0, 1, 0])

    print("[2] 증강·학습 데이터 변환 점검")
    frame = synth_frame(rng, True)
    boxes = np.array([[0, OBJ_ROI[0], OBJ_ROI[1] + 240, OBJ_ROI[2], OBJ_ROI[3] + 240]], np.float32)
    aug_img, aug_boxes = PhysicalAugmenter(seed=0)(frame, boxes)
    check("물리 열화 증강 출력 형태 유지", aug_img.shape == frame.shape and aug_img.dtype == np.uint8)
    cut = np.zeros((40, 60, 4), np.uint8)
    cv2.ellipse(cut, (30, 20), (28, 18), 0, 0, 360, (240, 240, 240, 255), -1)
    cp = CopyPaster([Cutout(cut, 1, 0.06)], geo, cfg.pre.roi_top, n_range=(2, 2), seed=0)
    _, cp_boxes = cp(to_linear(frame), boxes.copy())
    check(f"Copy-Paste 박스 추가 ({len(boxes)} → {len(cp_boxes)})", len(cp_boxes) > len(boxes))
    seam = np.array([[0, 630, 554, 650, 566]], np.float32)          # 겹침 영역(608~672) 안의 물체
    rb = boxes_frame_to_roi(seam, cfg.pre.roi_top, 480)
    tb = [boxes_roi_to_tile(rb, x0, 672, 480) for x0 in (0, 608)]
    check("겹침 영역 물체는 두 타일 모두에 라벨", all(len(t) == 1 for t in tb))

    print("[3] 온보드 파이프라인 (합성 영상 40프레임)")
    # 반사점 자리(ROI 640, 360) 주변 반지름 20px를 반사 마스크로 지정
    spec = np.zeros((480, 1280), bool)
    cv2.circle(spec.view(np.uint8), (640, 360), 20, 1, -1)
    fake = FakeDetector(cfg)
    pipe = OnboardPipeline(cfg, fake, flatfield=None, specular_mask=spec)
    led_on, confirmed, times, last_proc = False, [], [], None
    for i in range(40):
        f = synth_frame(rng, led_on)                  # 직전 명령대로 LED를 켜거나 끈 장면 생성
        fake.lit = led_on
        meta = FrameMeta(exposure_ms=4.0 if led_on else 33.0, gain=4.0 if led_on else 8.0, timestamp=i * 0.1)
        t0 = time.perf_counter()
        res = pipe.step(f, meta)
        times.append((time.perf_counter() - t0) * 1000)
        state = "확인용" if not res.led.use_frame else ("LED" if res.led.frame_led_on else "주변광")
        q = "-" if res.quality is None else (res.quality.reason or "ok")
        names = [CLASS_NAMES[int(c)] for c in res.detections[:, 5]] if len(res.detections) else []
        print(f"  #{i + 1:02d} {state:4s} 품질={q:11s} 검출={names} 확정={[CLASS_NAMES[t.cls] for t in res.confirmed]}")
        confirmed += res.confirmed
        if res.processed is not None and res.led.frame_led_on:
            last_proc = (f, res)
        led_on = res.led.led_on                       # 다음 프레임에 적용할 LED 상태
    check("에어팟이 정확히 1번 확정됨", [t.cls for t in confirmed] == [0])
    check("반사점·방해 물체는 확정되지 않음", len(confirmed) == 1)
    print(f"  프레임당 처리 시간(이 PC, 가짜 검출기): 중앙값 {np.median(times):.1f}ms")

    if last_proc is not None:
        os.makedirs(out_dir, exist_ok=True)
        f, res = last_proc
        path = os.path.join(out_dir, "selftest_compare.png")
        imwrite(path, visualize(f, res, cfg))
        print(f"  비교 이미지 저장: {path} (위: 원본 ROI / 아래: 전처리 결과)")
    print("셀프테스트 통과")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 녹화 영상·사진 데모                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def visualize(frame, res, cfg):
    """위: 원본 ROI / 아래: 전처리 결과 + 검출 박스(빨강) + 타일 경계(하늘색)."""
    top = cfg.pre.roi_top
    raw = frame[top:].copy()
    proc = res.processed.copy() if res.processed is not None else raw.copy()
    for x1, y1, x2, y2, conf, cls in res.detections:
        p1, p2 = (int(x1), int(y1) - top), (int(x2), int(y2) - top)   # 원본 좌표 → ROI 좌표
        cv2.rectangle(proc, p1, p2, (0, 0, 255), 2)
        cv2.putText(proc, f"{CLASS_NAMES[int(cls)]} {conf:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for x in tile_offsets(proc.shape[1], cfg.pre.tile_w, cfg.pre.tile_overlap)[1:]:
        cv2.line(proc, (x, 0), (x, proc.shape[0]), (255, 255, 0), 1)
    return np.vstack([raw, proc])


def iter_frames(args):
    """영상 또는 사진 폴더에서 (번호, 프레임, 시각)을 하나씩 꺼낸다."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield i, f, i / fps
            i += 1
    else:
        paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in glob.glob(os.path.join(args.images, ext)))
        for i, p in enumerate(paths):
            f = imread(p)
            if f is not None:
                yield i, f, i / 10.0


def demo(args):
    cfg = Config()
    detector = make_detector(args.weights, (cfg.pre.tile_h, cfg.pre.tile_w)) if args.weights else None
    pipe = OnboardPipeline(cfg, detector)
    # 녹화 영상에는 노출·게인 정보가 없으므로 LED 자동 판단 대신 녹화 당시 상태로 고정한다
    pipe.force_led = {"on": True, "off": False}[args.led]
    os.makedirs(args.out, exist_ok=True)
    print(f"플랫필드: {'있음' if pipe.pre.flatfield is not None else '없음'} / 반사 마스크: "
          f"{'있음' if pipe.spec_mask is not None else '없음'}")
    for i, f, t in iter_frames(args):
        if f.shape[:2] != (cfg.camera.height, cfg.camera.width):
            f = cv2.resize(f, (cfg.camera.width, cfg.camera.height))
            if i == 0:
                print("⚠ 해상도가 달라 1280×720으로 맞췄습니다 (실제 로봇 해상도로 찍은 영상 사용 권장)")
        res = pipe.step(f, FrameMeta(0, 0, t))
        for tr in res.confirmed:
            print(f"  ★ #{i} 확정: {CLASS_NAMES[tr.cls]} 위치 {tr.pos.round(3)} conf {tr.best_conf:.2f}")
        if res.quality is not None and not res.quality.ok:
            print(f"  #{i} 건너뜀: {res.quality.reason}")
        if i % args.every == 0:
            imwrite(os.path.join(args.out, f"frame_{i:05d}.png"), visualize(f, res, cfg))
    print(f"결과 저장: {args.out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 라즈베리 파이 실시간 실행                                                    ║
# ║    카메라 → 파이프라인 → LED·노출 명령 적용 을 무한 반복한다.                   ║
# ║    ROS 2 노드로 옮길 때도 이 루프 구조를 그대로 쓰면 된다.                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def live(args):
    from rpi_hardware import LedPwm, RpiCamera

    cfg = Config()
    detector = make_detector(args.weights, (cfg.pre.tile_h, cfg.pre.tile_w)) if args.weights else None
    pipe = OnboardPipeline(cfg, detector)
    cam = RpiCamera(cfg)
    led = None if args.no_led else LedPwm(cfg)
    led_on, times = False, []
    print("실행 중 (Ctrl+C로 종료)")
    try:
        while True:
            frame, meta, _ = cam.capture()
            t0 = time.perf_counter()
            res = pipe.step(frame, meta)
            times.append((time.perf_counter() - t0) * 1000)

            # 파이프라인이 정한 명령을 하드웨어에 적용 (다음 프레임부터 반영됨)
            c = res.led
            settle = 0
            if c.led_on != led_on:
                if led is not None:
                    led.set(c.led_on)
                led_on = c.led_on
                settle = cfg.rpi.settle_frames_led
            if c.fixed_exposure != cam.fixed:
                cam.set_mode(c.fixed_exposure)
                settle = max(settle, cfg.rpi.settle_frames_exposure)
            if settle:
                cam.drop_frames(settle)              # 설정이 반쯤 적용된 프레임은 버린다
            if res.quality is not None and res.quality.reason == "overexposed" and led is not None:
                led.dim()                             # 과노출이면 LED를 한 단계 어둡게

            for tr in res.confirmed:
                print(f"★ 확정: {CLASS_NAMES[tr.cls]} 위치 {tr.pos.round(3)} conf {tr.best_conf:.2f}")
            if len(times) % 50 == 0:
                print(f"  처리 시간 중앙값 {np.median(times[-50:]):.1f}ms / LED {'ON' if led_on else 'OFF'}")
    except KeyboardInterrupt:
        pass
    finally:
        if led is not None:
            led.close()
        cam.close()
        if detector is not None and hasattr(detector, "close"):
            detector.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--images")
    ap.add_argument("--led", choices=["on", "off"], default="on", help="녹화 당시 LED 상태")
    ap.add_argument("--live", action="store_true", help="Raspberry Pi 카메라로 실시간 실행")
    ap.add_argument("--no-led", action="store_true", help="--live에서 LED PWM을 쓰지 않음")
    ap.add_argument("--weights", default=None, help="모델: .hef(Hailo) 또는 .pt/.onnx(PC 검증용). 없으면 전처리만")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--every", type=int, default=10, help="N프레임마다 결과 이미지 저장")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.out)
    elif args.live:
        live(args)
    elif args.video or args.images:
        demo(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
