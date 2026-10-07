"""PC에서 하는 오프라인 작업 모음: 캘리브레이션 · 중복 제거 · 증강 · 학습 데이터셋 · Hailo 보정 세트

================================================================================
이 파일만 따로 둔 이유
    여기 있는 작업은 로봇이 돌아다니는 동안이 아니라 **미리 PC에서 한 번씩** 하는 일이다.
    로봇(Pi)에 올라갈 필요가 없고, 로봇 실시간 코드(preprocess.py)를 읽을 때 방해가 되지
    않도록 분리했다. 대신 영상 전처리 자체는 preprocess.py의 것을 그대로 가져다 쓴다
    (학습 데이터와 로봇 입력이 똑같은 전처리를 거치게 하려는 것이 핵심).

전체 작업 순서 (각 단계가 아래 명령 하나)
    1. calibrate horizon    카메라 자세로 지평선 위치·물체 픽셀 크기 확인
    2. calibrate flatfield  LED 조명 얼룩 보정 맵 만들기       → calib/flatfield_led.npy
    3. calibrate specular   유광 바닥 LED 반사점 마스크 만들기  → calib/specular_mask.npy
    4. dedup                영상에서 뽑은 비슷한 프레임 지우기
    5. build                증강 + 전처리 + 타일링 → YOLO 학습 데이터셋
    6. calib-set            Hailo INT8 변환용 보정 이미지 고르기

사용 예
    python offline_tools.py calibrate horizon
    python offline_tools.py calibrate flatfield --images calib_raw/flat --preview
    python offline_tools.py calibrate specular  --images calib_raw/glare --preview
    python offline_tools.py dedup     --images raw/s01/images --labels raw/s01/labels --out dedup/s01
    python offline_tools.py build     --images raw/train/images --labels raw/train/labels --out ds/train --variants 3 --cutouts cutouts
    python offline_tools.py build     --images raw/val/images   --labels raw/val/labels   --out ds/val   --variants 0
    python offline_tools.py calib-set --tiles ds/train/images --out calib_set --n 1024
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from preprocess import (GLARE_CLASS, CameraGeometry, Config, Preprocessor, boxes_frame_to_roi,
                        boxes_roi_to_tile, imread, imwrite, load_flatfield, split_tiles,
                        to_linear, to_srgb)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def list_images(folder):
    """폴더 안 이미지 파일 경로를 이름순으로. (영상에서 뽑은 프레임이면 이름순 = 시간순)"""
    return sorted(p for ext in IMG_EXTS for p in glob.glob(os.path.join(folder, ext)))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 1. 현장 캘리브레이션                                                         ║
# ║    로봇 카메라로 찍은 사진으로 보정값을 만든다. 사진은 반드시 로봇의 LED 모드와   ║
# ║    같은 설정(노출 4ms, 게인·화이트밸런스 고정, 샤프닝 0, 같은 초점)으로 찍는다.  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

OBJECTS = [("귀걸이", 0.008), ("반지", 0.02), ("이어버드", 0.03), ("열쇠", 0.05),
           ("케이스", 0.06), ("리모컨", 0.15)]


def read_calib_images(folder, cfg):
    imgs = [im for im in (imread(p) for p in list_images(folder)) if im is not None]
    if not imgs:
        sys.exit(f"이미지를 찾지 못했습니다: {folder}")
    h, w = imgs[0].shape[:2]
    if (w, h) != (cfg.camera.width, cfg.camera.height):
        sys.exit(f"해상도 {w}x{h} != 설정 {cfg.camera.width}x{cfg.camera.height}")
    return imgs


def calibrate_horizon(cfg, args):
    """카메라 자세(높이 5cm, 15° 하향)로 지평선 행을 계산해 ROI 크롭 값이 적절한지 확인하고,
    거리별 물체 픽셀 크기 표를 출력한다. 사진이 필요 없다(설정값만으로 계산)."""
    geo = CameraGeometry(cfg.camera)
    hr = geo.horizon_row()
    roi_top_for_tile = cfg.camera.height - cfg.pre.tile_h    # 타일 높이(480)를 맞추려면 위에서 몇 행을 잘라야 하나
    print(f"지평선 행(핀홀 근사): {hr:.1f}")
    print(f"현재 roi_top: {cfg.pre.roi_top}  /  타일 높이를 맞추는 roi_top: {roi_top_for_tile}")
    if roi_top_for_tile > hr:
        # 지평선보다 아래를 자르면 먼 바닥 일부가 잘린다. 얼마나 먼 곳부터 잘리는지 계산
        far, ok = geo.pixel_to_floor([[cfg.camera.cx, roi_top_for_tile]])
        print(f"ROI 맨 윗줄은 약 {far[0, 0]:.2f}m 앞 바닥입니다. 그보다 먼 바닥은 잘립니다.")
        if not ok[0] or far[0, 0] < 1.2:
            print("⚠ 탐색 거리가 짧습니다. tile_h를 키우거나 카메라 하향 각도를 줄이는 것을 검토하세요.")
    else:
        print(f"✓ 지평선 위 {hr - roi_top_for_tile:.0f}px 여유를 두고 잘립니다.")
    print("\n거리별 물체 픽셀 크기 (가장 긴 변, 근사)  ← 10px 미만이면 검출이 어렵다")
    dists = (0.3, 0.5, 1.0, 1.5)
    print("물체".ljust(8) + "".join(f"{d:>8.1f}m" for d in dists))
    for name, size in OBJECTS:
        print(name.ljust(8) + "".join(f"{cfg.camera.fx * size / d:>8.1f}px" for d in dists))


def calibrate_flatfield(cfg, args):
    """LED 조명 얼룩 보정 맵 만들기.

    준비: 어두운 곳에서 LED를 켜고 무광 회색 시트(무늬 없는 평평한 면)를 10~30장 찍는다.
    원리: 회색 시트는 어디나 반사율이 같으므로, 사진의 밝기 차이 = 순수한 조명 얼룩이다.
          게인 = (화면 중앙 밝기) / (각 위치 밝기) 를 곱하면 어디나 중앙만큼 밝아진다.
    """
    imgs = read_calib_images(args.images, cfg)
    top = cfg.pre.roi_top
    # 1) 여러 장을 선형 공간에서 평균 → 센서 노이즈가 줄어든다
    acc = np.zeros(imgs[0][top:].shape, np.float64)
    for im in imgs:
        acc += to_linear(im[top:])
    mean = (acc / len(imgs)).astype(np.float32)
    # 2) 색 채널을 합쳐 밝기(휘도) 하나로. 계수는 사람 눈 감도 기준 (B 0.114, G 0.587, R 0.299)
    lum = 0.114 * mean[..., 0] + 0.587 * mean[..., 1] + 0.299 * mean[..., 2]
    # 3) 크게 흐리게 → 시트의 미세한 결은 지우고 넓게 퍼진 조명 분포만 남긴다
    smooth = cv2.GaussianBlur(lum, (0, 0), args.sigma)
    # 4) 화면 중앙 20% 영역 밝기를 기준(ref)으로 각 위치의 게인을 계산
    h, w = smooth.shape
    ref = float(np.median(smooth[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]))
    gain = np.clip(ref / np.maximum(smooth, 1e-4), cfg.pre.ff_gain_min, cfg.pre.ff_gain_max)
    gain = gain.astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, gain)
    sat = float((mean.max(axis=2) >= 0.98).mean())
    print(f"저장: {args.out}  shape={gain.shape}  게인 범위 {gain.min():.2f}~{gain.max():.2f}")
    print(f"게인 상한({cfg.pre.ff_gain_max})에 걸린 비율: {(gain >= cfg.pre.ff_gain_max - 1e-3).mean():.1%}")
    if sat > 0.01:
        print(f"⚠ 평균 영상의 {sat:.1%}가 포화 상태입니다. 포화는 보정할 수 없으니 LED 밝기·각도를 먼저 조정하세요.")
    if args.preview:     # 게인 맵을 색으로 표시 (파랑 = 게인 작음, 빨강 = 게인 큼)
        vis = cv2.applyColorMap(np.uint8(255 * (gain - gain.min()) / (np.ptp(gain) + 1e-6)), cv2.COLORMAP_JET)
        imwrite(os.path.splitext(args.out)[0] + "_preview.png", vis)


def calibrate_specular(cfg, args):
    """유광 바닥의 LED 반사점 마스크 만들기.

    준비: 어두운 곳에서 LED를 켜고 유광 타일 위를 주행하며 10~30장 찍는다.
    원리: 카메라·LED·바닥이 고정된 기하라서 반사점은 화면 속 늘 같은 자리에 생긴다.
          여러 장 중 절반 이상에서 하얗게 날아간 픽셀 = 반사점 자리.
    """
    imgs = read_calib_images(args.images, cfg)
    top = cfg.pre.roi_top
    frac = np.mean([im[top:].max(axis=2) >= args.thr for im in imgs], axis=0)   # 픽셀별 포화 빈도
    mask = (frac >= args.min_frac).astype(np.uint8)
    # 마스크를 조금 키운다(팽창): 줄눈 위를 지날 때 진동으로 반사점이 약간씩 흔들리기 때문
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1))
    mask = cv2.dilate(mask, k) > 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, mask)
    print(f"저장: {args.out}  shape={mask.shape}  마스크 면적 {mask.mean():.2%}")
    if mask.mean() > 0.05:
        print("⚠ 마스크가 화면의 5%를 넘습니다. LED 확산판·교차 편광으로 반사부터 줄이는 것을 권장합니다.")
    if args.preview:     # 반사점 자리를 빨갛게 칠한 사진
        vis = imgs[0][top:].copy()
        vis[mask] = (0.5 * vis[mask] + [0, 0, 127]).astype(np.uint8)
        imwrite(os.path.splitext(args.out)[0] + "_preview.png", vis)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 2. 중복 프레임 제거 (pHash)                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  30fps 영상에서 프레임을 다 뽑으면 거의 똑같은 사진이 수십 장 생긴다. 그대로 학습하면
#  같은 장면을 외워 버리고(과적합), 검증 세트에 비슷한 사진이 섞이면 성능이 부풀려진다.
#
#  pHash(지각 해시): 사진을 64비트 "지문"으로 요약. 비슷한 사진은 지문도 비슷하다.
#  두 지문이 다른 비트 수(해밍 거리)가 6 이하면 같은 사진으로 본다.

def phash(img) -> np.ndarray:
    """사진 → 64비트 지문 (8바이트 배열)."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))[:8, :8].flatten()    # 주파수 변환 후 저주파(큰 윤곽) 64개만
    return np.packbits(d > np.median(d[1:]))               # 중앙값보다 크면 1, 작으면 0


def hamming(a, b) -> int:
    """두 지문에서 서로 다른 비트 수."""
    return int(np.unpackbits(a ^ b).sum())


def run_dedup(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    if args.labels:
        os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    paths = list_images(args.images)
    recent = deque(maxlen=args.window)        # 최근에 남긴 사진 지문 50개와만 비교 (전부 비교하면 느림)
    kept = 0
    for p in paths:
        img = imread(p)
        if img is None:
            continue
        h = phash(img)
        if any(hamming(h, r) <= args.thr for r in recent):
            continue                          # 최근 사진과 너무 비슷하면 건너뜀
        recent.append(h)
        kept += 1
        shutil.copy2(p, os.path.join(args.out, "images"))
        if args.labels:
            lbl = os.path.join(args.labels, os.path.splitext(os.path.basename(p))[0] + ".txt")
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(args.out, "labels"))
    print(f"{len(paths)}장 중 {kept}장 유지 ({len(paths) - kept}장 중복 제거)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 3. 학습용 증강                                                               ║
# ║    모두 "선형 공간의 원본 프레임"에서, 전처리(Preprocessor) "이전에" 적용한다.  ║
# ║    로봇에서는 이런 현상이 촬영 순간에 먼저 생기고 전처리가 나중에 적용되므로,    ║
# ║    학습도 같은 순서여야 로봇이 실제로 보는 사진과 분포가 같아진다.              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  박스 형식: (N,5) float32 [클래스, x1, y1, x2, y2] 원본 프레임 픽셀 좌표
#
#  적용 순서 = 빛이 센서에 닿기까지의 물리 순서
#     Copy-Paste(장면에 물건 추가) → 조명 색·밝기 → LED 반사점 → 비네팅 → 렌즈 뿌연 막
#     → 모션블러 → 렌즈에 붙은 털 → 센서 노이즈 → 감마 인코딩(sRGB)

@dataclass
class AugConfig:
    """각 증강의 적용 확률(p_)과 세기 범위. 범위는 (최소, 최대)에서 무작위로 고른다."""
    p_illum: float = 0.5
    brightness: tuple = (0.7, 1.3)        # 밝기 ±30%
    p_glare: float = 0.3
    glare_count: tuple = (1, 3)
    glare_sigma: tuple = (2.0, 6.0)       # 반사점 크기(px)
    glare_amp: tuple = (2.0, 8.0)         # 반사점 밝기 (선형 값, 1 이상이면 하얗게 날아감)
    label_glare: bool = True              # 가짜 반사점을 led_glare 클래스로 라벨링 → 모델이 "이건 반사"라고 배움
    p_vignette: float = 0.4
    vignette_strength: tuple = (0.2, 0.5)
    p_haze: float = 0.2
    haze_alpha: tuple = (0.05, 0.2)
    p_blur: float = 0.3
    blur_len: tuple = (3, 7)              # 번짐 길이(px)
    p_hair: float = 0.2
    p_noise: float = 0.5
    photon_scale: tuple = (50.0, 400.0)   # 작을수록 어두운 환경(노이즈 큼)
    read_sigma: tuple = (0.002, 0.01)


class PhysicalAugmenter:
    """실제로 일어나는 화질 저하(조명·반사·번짐·노이즈 등)를 흉내 내 학습 사진을 늘린다."""

    def __init__(self, cfg: AugConfig = None, seed=None):
        self.cfg = cfg or AugConfig()
        self.rng = np.random.default_rng(seed)     # seed를 고정하면 매번 같은 결과 → 재현 가능

    def __call__(self, img_u8: np.ndarray, boxes: np.ndarray):
        """uint8 사진 입력용 편의 함수."""
        lin, boxes = self.apply_linear(to_linear(img_u8), boxes)
        return to_srgb(lin), boxes

    def apply_linear(self, lin: np.ndarray, boxes: np.ndarray):
        """선형 공간 사진(0~1 float)에 확률적으로 증강을 적용. (사진, 박스) 반환."""
        c, r = self.cfg, self.rng
        lin = lin.astype(np.float32, copy=True)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 5)
        if r.random() < c.p_illum:
            lin = self._illumination(lin)
        if r.random() < c.p_glare:
            lin, gb = self._glare(lin)
            if c.label_glare and len(gb):
                boxes = np.vstack([boxes, gb])
        if r.random() < c.p_vignette:
            lin = lin * self._vignette(lin.shape[:2])[..., None]
        if r.random() < c.p_haze:                  # 렌즈 뿌연 막: 전체를 평균 밝기 쪽으로 살짝 섞음 → 대비 저하
            a = r.uniform(*c.haze_alpha)
            lin = lin * (1 - a) + a * float(lin.mean())
        if r.random() < c.p_blur:
            lin = self._motion_blur(lin)
        if r.random() < c.p_hair:
            lin = self._hair(lin)
        if r.random() < c.p_noise:
            lin = self._noise(lin)
        return lin, boxes

    def _illumination(self, lin):
        """밝기 ±30% + 색온도(전구색 2700K ~ 주광색 6500K) 흉내. 채널 순서는 B, G, R."""
        r = self.rng
        t = r.uniform(0, 1)                                  # 0 = 노란 전구색, 1 = 푸른 주광색
        warm, cool = np.array([0.75, 1.0, 1.15]), np.array([1.15, 1.0, 0.9])
        gains = (warm * (1 - t) + cool * t) * r.uniform(*self.cfg.brightness)
        return lin * gains.astype(np.float32)

    def _glare(self, lin):
        """유광 바닥에 비친 LED 같은 하얀 점을 1~3개 그린다. 그 위치에 led_glare 박스도 만든다."""
        r, c = self.rng, self.cfg
        h, w = lin.shape[:2]
        out, boxes = lin, []
        for _ in range(r.integers(c.glare_count[0], c.glare_count[1] + 1)):
            s = r.uniform(*c.glare_sigma)
            cx, cy = r.uniform(0, w), r.uniform(h * 0.4, h)          # 반사점은 화면 아래쪽(바닥)에 생김
            rad = int(math.ceil(3 * s))                               # 가우시안은 3σ 밖이면 거의 0
            x0, x1 = max(int(cx) - rad, 0), min(int(cx) + rad + 1, w)
            y0, y1 = max(int(cy) - rad, 0), min(int(cy) + rad + 1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            blob = r.uniform(*c.glare_amp) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
            out[y0:y1, x0:x1] += blob[..., None].astype(np.float32)
            boxes.append([GLARE_CLASS, max(cx - 2 * s, 0), max(cy - 2 * s, 0),
                          min(cx + 2 * s, w), min(cy + 2 * s, h)])
        return out, np.array(boxes, np.float32).reshape(-1, 5)

    def _vignette(self, hw):
        """비네팅: 가운데가 밝고 가장자리로 갈수록 어두워지는 배율 맵. LED는 화면 아래쪽 중심을 비춘다."""
        r = self.rng
        h, w = hw
        cx, cy = w / 2 + r.uniform(-0.1, 0.1) * w, h * r.uniform(0.6, 1.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d2 = ((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h * 0.8)) ** 2   # 중심에서의 거리²
        return np.clip(1.0 - r.uniform(*self.cfg.vignette_strength) * d2, 0.05, 1.0)

    def _motion_blur(self, lin):
        """주행 중 번짐: 짧은 선분 모양 커널로 사진을 문지른다. 70%는 거의 수평(제자리 회전 시 번짐)."""
        r = self.rng
        L = int(r.integers(self.cfg.blur_len[0], self.cfg.blur_len[1] + 1)) | 1   # | 1 → 홀수로 (중심이 있도록)
        ang = r.uniform(-10, 10) if r.random() < 0.7 else r.uniform(0, 180)
        k = np.zeros((L, L), np.float32)
        c = L // 2
        dx, dy = math.cos(math.radians(ang)) * c, math.sin(math.radians(ang)) * c
        cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1)
        return cv2.filter2D(lin, -1, k / k.sum())          # 합이 1이 되게 나눠서 밝기는 유지

    def _hair(self, lin):
        """렌즈에 붙은 고양이 털: 초점이 안 맞아 흐릿한, 가늘고 어두운 곡선 1~4개."""
        r = self.rng
        h, w = lin.shape[:2]
        mask = np.zeros((h, w), np.float32)
        for _ in range(r.integers(1, 5)):
            p = np.array([r.uniform(0, w), r.uniform(0, h)])
            pts = [p.copy()]
            ang = r.uniform(0, 2 * math.pi)
            for _ in range(6):                             # 방향을 조금씩 틀며 6마디 이어 그림 → 구불구불
                ang += r.uniform(-0.6, 0.6)
                p = p + r.uniform(20, 80) * np.array([math.cos(ang), math.sin(ang)])
                pts.append(p.copy())
            cv2.polylines(mask, [np.int32(pts)], False, 1.0, int(r.integers(1, 4)))
        mask = cv2.GaussianBlur(mask, (0, 0), r.uniform(2, 6))   # 렌즈에 붙어 있으니 초점이 안 맞아 흐림
        return lin * (1 - np.clip(mask * r.uniform(0.4, 0.9), 0, 1))[..., None]

    def _noise(self, lin):
        """센서 노이즈. 빛 알갱이 수의 무작위성(Poisson) + 회로 잡음(Gaussian).

        photon_scale이 작을수록 픽셀당 빛 알갱이가 적다 = 어두운 환경 = 노이즈가 상대적으로 크다.
        """
        r = self.rng
        k = r.uniform(*self.cfg.photon_scale)
        noisy = r.poisson(np.clip(lin, 0, None) * k).astype(np.float32) / k
        return noisy + r.normal(0, r.uniform(*self.cfg.read_sigma), lin.shape).astype(np.float32)


# Copy-Paste 증강
# ----------------------------------------------------------------------------
#  배경을 투명하게 뺀 물건 사진(RGBA PNG)을 바닥 사진에 붙여 학습 데이터를 늘린다.
#  작은 물건 사진을 많이 모으기 어려울 때 가장 효과적인 방법이다.
#  핵심: 아무 크기로 붙이면 안 되고, 붙일 위치(화면 높이)에 맞는 "실제 크기"로 붙여야 한다.
#        화면 아래쪽(가까움)이면 크게, 위쪽(멂)이면 작게 → CameraGeometry.object_size_px 사용.

@dataclass
class Cutout:
    rgba: np.ndarray     # uint8 (H,W,4). 4번째 채널(alpha) 0 = 투명 배경
    cls: int             # 클래스 번호
    size_m: float        # 실제 가장 긴 변 길이 (m)


def load_cutouts(folder: str):
    """물건 사진 폴더 읽기. 파일명 규칙: <클래스>_<크기cm>_<아무거나>.png  예) 2_2.0_ring01.png"""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
        parts = os.path.basename(path).split("_")
        img = imread(path, cv2.IMREAD_UNCHANGED)        # UNCHANGED: 투명(alpha) 채널까지 읽음
        if img is None or img.ndim != 3 or img.shape[2] != 4 or len(parts) < 3:
            continue
        out.append(Cutout(img, int(parts[0]), float(parts[1]) / 100.0))
    return out


class CopyPaster:
    """선형 공간 사진에 물건을 1~4개 붙이고, 붙인 자리에 박스를 추가한다."""

    def __init__(self, cutouts, geometry: CameraGeometry, roi_top: int,
                 n_range=(1, 4), shadow=0.35, seed=None):
        self.cutouts = cutouts
        self.geo = geometry
        self.roi_top = roi_top
        self.n_range = n_range
        self.shadow = shadow             # 그림자 진하기 (0~1)
        self.rng = np.random.default_rng(seed)

    def __call__(self, lin: np.ndarray, boxes: np.ndarray):
        if not self.cutouts:
            return lin, boxes
        r = self.rng
        h, w = lin.shape[:2]
        v_min = max(self.roi_top, self.geo.horizon_row() + 30)   # 지평선 근처는 너무 작아서 제외
        new = []
        for _ in range(r.integers(self.n_range[0], self.n_range[1] + 1)):
            obj = self.cutouts[r.integers(len(self.cutouts))]
            for _try in range(10):                # 3px 이상으로 보이는 위치를 최대 10번 시도
                u, v = r.uniform(20, w - 20), r.uniform(v_min, h - 5)
                size_px = float(self.geo.object_size_px([v], [u], obj.size_m)[0])
                if np.isfinite(size_px) and size_px >= 3:
                    break
            else:
                continue
            patch = self._prepare(obj, size_px, r.uniform(0, 360))
            self._shadow(lin, patch, u, v)       # 그림자 먼저, 그 위에 물건
            bbox = _paste(lin, patch, u, v)
            if bbox is not None:
                new.append([obj.cls, *bbox])
        if new:
            boxes = np.vstack([np.asarray(boxes, np.float32).reshape(-1, 5), np.array(new, np.float32)])
        return lin, boxes

    @staticmethod
    def _prepare(obj: Cutout, size_px: float, angle: float) -> np.ndarray:
        """물건 사진을 회전 → 목표 크기로 축소 → [선형 RGB, alpha(0~1)] float 패치로."""
        rgba = obj.rgba
        ph, pw = rgba.shape[:2]
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        # 회전하면 모서리가 잘리므로 캔버스를 키운다
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(ph * sin + pw * cos), int(ph * cos + pw * sin)
        M[0, 2] += nw / 2 - pw / 2
        M[1, 2] += nh / 2 - ph / 2
        rot = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        s = size_px / max(ph, pw)                 # 원래 사진의 긴 변이 size_px가 되도록
        tw, th = max(int(round(nw * s)), 1), max(int(round(nh * s)), 1)
        rot = cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out = np.empty(rot.shape, np.float32)
        out[..., :3] = to_linear(rot[..., :3])    # 바닥 사진과 같은 선형 공간으로 맞춘다
        out[..., 3] = rot[..., 3] / 255.0
        return out

    def _shadow(self, lin, patch, u, v):
        """물건 모양을 아래로 살짝 옮기고 흐리게 해서 바닥에 어두운 그림자를 깐다."""
        ph, pw = patch.shape[:2]
        a = patch[..., 3]
        pad = max(ph, pw)
        m = np.zeros((ph + 2 * pad, pw + 2 * pad), np.float32)
        dy = int(ph * 0.15)
        m[pad + dy:pad + dy + ph, pad:pad + pw] = a
        m = cv2.GaussianBlur(m, (0, 0), max(pw, ph) * 0.15 + 0.5)
        sh = np.zeros(m.shape + (4,), np.float32)     # 색은 검정, alpha만 그림자 모양
        sh[..., 3] = m * self.shadow
        _paste(lin, sh, u, v)


def _paste(lin: np.ndarray, patch: np.ndarray, cx: float, cy: float):
    """patch를 (cx, cy) 중심에 alpha 합성:  결과 = 바닥×(1-α) + 물건×α
    화면 밖으로 나가는 부분은 잘라 낸다. 실제로 붙은 영역의 박스 (x1,y1,x2,y2)를 반환."""
    H, W = lin.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + pw, W), min(y0 + ph, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    p = patch[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    a = p[..., 3:4]
    lin[iy0:iy1, ix0:ix1] = lin[iy0:iy1, ix0:ix1] * (1 - a) + p[..., :3] * a
    ys, xs = np.where(p[..., 3] > 0.5)               # 반 이상 불투명한 픽셀로 박스를 정한다
    if len(xs) == 0:
        return None
    return (ix0 + xs.min(), iy0 + ys.min(), ix0 + xs.max() + 1, iy0 + ys.max() + 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 4. 학습 데이터셋 만들기                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  원본 프레임 1장 → [원본 v0 + 증강 사본 a1, a2, ...] 각각
#      → 전처리(로봇과 같은 코드) → 박스를 ROI·타일 좌표로 변환 → 타일 2장 + YOLO 라벨 저장
#
#  YOLO 라벨 형식(txt 한 줄 = 박스 하나): 클래스 중심x 중심y 너비 높이  (모두 0~1 비율)
#  LED를 켜고 찍은 사진은 파일명에 "_led"를 넣어 LED 모드 전처리(플랫필드)를 적용한다.
#  train/val/test 분할은 이 단계 전에 "세션·장소 단위"로 끝내야 한다 (같은 장면이 섞이지 않게).

def read_yolo(path, w, h) -> np.ndarray:
    """YOLO 라벨 txt → (N,5) 픽셀 박스. 파일이 없으면 빈 배열(물건 없는 배경 사진)."""
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float32)
    rows = [list(map(float, ln.split())) for ln in open(path, encoding="utf-8") if ln.strip()]
    out = []
    for c, cx, cy, bw, bh in rows:
        out.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, np.float32).reshape(-1, 5)


def write_yolo(path, boxes, w, h):
    """(N,5) 픽셀 박스 → YOLO 라벨 txt."""
    with open(path, "w", encoding="utf-8") as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")


def erase_tiny(img, boxes, min_px):
    """너무 작은 물체(4px 미만)는 라벨을 지우는 대신 그 자리를 주변 픽셀로 메운다(inpaint).

    라벨만 지우고 물체를 사진에 남겨 두면, 모델은 "여기 에어팟이 있는데 정답은 '없음'"을
    배우게 되어 오히려 해롭다. 아예 사진에서 지워 버리면 그런 모순이 없다.
    """
    if len(boxes) == 0:
        return img, boxes
    size = np.minimum(boxes[:, 3] - boxes[:, 1], boxes[:, 4] - boxes[:, 2])
    tiny = size < min_px
    if not tiny.any():
        return img, boxes
    mask = np.zeros(img.shape[:2], np.uint8)
    for _, x1, y1, x2, y2 in boxes[tiny]:
        cv2.rectangle(mask, (int(x1) - 1, int(y1) - 1), (int(np.ceil(x2)) + 1, int(np.ceil(y2)) + 1), 255, -1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA), boxes[~tiny]


def run_build(args):
    cfg = Config()
    W, H = cfg.camera.width, cfg.camera.height
    roi_h = cfg.roi_shape()[0]
    ff = load_flatfield(args.flatfield or cfg.flatfield_path, cfg.roi_shape(),
                        cfg.pre.ff_gain_min, cfg.pre.ff_gain_max)
    pre = Preprocessor(cfg.pre, ff)                       # ★ 로봇과 같은 전처리 객체
    aug = PhysicalAugmenter(AugConfig(), seed=args.seed)
    cutouts = load_cutouts(args.cutouts) if args.cutouts else []
    cp = CopyPaster(cutouts, CameraGeometry(cfg.camera), cfg.pre.roi_top, seed=args.seed + 1) if cutouts else None

    img_dir, lbl_dir = os.path.join(args.out, "images"), os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in glob.glob(os.path.join(args.images, ext)))
    warned_ff = False
    n_tiles = n_boxes = 0
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = imread(path)
        if img is None or img.shape[:2] != (H, W):
            print(f"건너뜀(해상도 불일치 또는 읽기 실패): {path}")
            continue
        led = args.led_tag in stem
        if led and ff is None and not warned_ff:
            print("⚠ 플랫필드 맵이 없어 LED 이미지도 플랫필드 없이 처리합니다 (로봇 설정과 같아야 함).")
            warned_ff = True
        boxes = read_yolo(os.path.join(args.labels, stem + ".txt"), W, H) if args.labels else np.zeros((0, 5), np.float32)

        # 1) 원본(v0) + 증강 사본(a1, a2, ...) 만들기
        variants = [("v0", img, boxes)]
        for k in range(args.variants):
            lin = to_linear(img)
            b = boxes.copy()
            if cp is not None:
                lin, b = cp(lin, b)                   # 물건 먼저 붙이고 (붙인 물건도 아래 열화를 똑같이 받도록)
            lin, b = aug.apply_linear(lin, b)          # 물리 열화: 반드시 전처리 '이전'
            variants.append((f"a{k + 1}", to_srgb(lin), b))

        # 2) 각각 전처리 → 박스 변환 → 타일로 잘라 저장
        for tag, im, b in variants:
            proc = pre(im, led)                        # 로봇과 같은 CPU 전처리 코드
            rb = boxes_frame_to_roi(b, cfg.pre.roi_top, roi_h)
            proc, rb = erase_tiny(proc, rb, cfg.post.min_box_px)
            tiles, xs = split_tiles(proc, cfg.pre)
            for i, (tile, x0) in enumerate(zip(tiles, xs)):
                tb = boxes_roi_to_tile(rb, x0, cfg.pre.tile_w, cfg.pre.tile_h)
                name = f"{stem}_{tag}_t{i}"            # 예) s01_000_led_a1_t0
                imwrite(os.path.join(img_dir, name + ".png"), tile)
                write_yolo(os.path.join(lbl_dir, name + ".txt"), tb, cfg.pre.tile_w, cfg.pre.tile_h)
                n_tiles += 1
                n_boxes += len(tb)
    print(f"완료: 원본 {len(paths)}장 → 타일 {n_tiles}장, 박스 {n_boxes}개 → {args.out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 5. Hailo INT8 보정 세트                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  Hailo는 모델을 INT8(정수 8비트)로 바꿔 빠르게 돌린다. 이때 "값이 보통 어느 범위에 있는지"를
#  예시 이미지로 재서 정한다(보정, calibration). 예시가 실제 로봇 입력과 다르면 범위가 틀어져
#  정확도가 떨어진다. 그래서:
#    - 전처리를 마친 타일(로봇이 Hailo에 넣는 것과 같은 것)에서 고르고
#    - 증강 안 한 원본(_v0_)을 우선 쓰고
#    - 어두운 LED 모드 타일을 30% 이상 넣는다 (빠지면 소파 밑 정확도가 크게 떨어짐)

def run_calib_set(args):
    rng = np.random.default_rng(args.seed)
    paths = sorted(glob.glob(os.path.join(args.tiles, "*.png")))
    if not paths:
        raise SystemExit(f"타일이 없습니다: {args.tiles}")

    def pick(pool, k):
        """pool에서 k장 무작위로, 원본(_v0_)을 먼저 채우고 모자라면 증강본으로."""
        pool = list(pool)
        rng.shuffle(pool)
        orig = [p for p in pool if "_v0_" in os.path.basename(p)]
        aug = [p for p in pool if "_v0_" not in os.path.basename(p)]
        return (orig + aug)[:k]

    led = [p for p in paths if args.led_tag in os.path.basename(p)]
    amb = [p for p in paths if args.led_tag not in os.path.basename(p)]
    n_led = min(len(led), max(int(round(args.n * args.led_ratio)), args.n - len(amb)))
    chosen = pick(led, n_led) + pick(amb, args.n - n_led)
    if len(chosen) < args.n:
        print(f"⚠ 타일이 {len(chosen)}장뿐입니다 (요청 {args.n}장).")
    if led and n_led / max(len(chosen), 1) < args.led_ratio:
        print(f"⚠ LED 모드 비율이 {n_led / len(chosen):.0%}로 목표 {args.led_ratio:.0%}보다 낮습니다.")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    arr = []
    for p in chosen:
        shutil.copy2(p, img_dir)                                 # hailomz --calib-path 용 폴더
        arr.append(cv2.cvtColor(imread(p), cv2.COLOR_BGR2RGB))   # Hailo 입력은 RGB
    np.save(os.path.join(args.out, "calib_set.npy"), np.stack(arr))   # DFC Python API 용 (N,480,672,3)
    print(f"보정 세트 {len(chosen)}장 (LED 모드 {n_led}장) → {args.out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 명령줄 진입점                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cal = sub.add_parser("calibrate", help="현장 캘리브레이션")
    csub = cal.add_subparsers(dest="what", required=True)
    csub.add_parser("horizon", help="지평선 위치와 거리별 물체 픽셀 크기")
    f = csub.add_parser("flatfield", help="LED 조명 얼룩 보정 맵")
    f.add_argument("--images", required=True)
    f.add_argument("--out", default="calib/flatfield_led.npy")
    f.add_argument("--sigma", type=float, default=25.0, help="조명 분포를 뽑을 때 흐리는 정도")
    f.add_argument("--preview", action="store_true")
    s = csub.add_parser("specular", help="LED 반사점 마스크")
    s.add_argument("--images", required=True)
    s.add_argument("--out", default="calib/specular_mask.npy")
    s.add_argument("--thr", type=int, default=250)
    s.add_argument("--min-frac", type=float, default=0.5, help="이 비율 이상의 사진에서 포화된 픽셀을 반사점으로 판단")
    s.add_argument("--dilate", type=int, default=15, help="마스크를 키우는 폭(px)")
    s.add_argument("--preview", action="store_true")

    d = sub.add_parser("dedup", help="연속 프레임 중복 제거")
    d.add_argument("--images", required=True)
    d.add_argument("--labels", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--thr", type=int, default=6, help="해밍 거리 이하면 중복")
    d.add_argument("--window", type=int, default=50)

    b = sub.add_parser("build", help="증강·전처리·타일링된 학습 데이터셋 생성")
    b.add_argument("--images", required=True, help="원본 프레임(1280×720) 폴더")
    b.add_argument("--labels", default=None, help="원본 프레임 기준 YOLO 라벨 폴더")
    b.add_argument("--out", required=True)
    b.add_argument("--variants", type=int, default=3, help="원본 1장당 증강 사본 수 (val/test는 0)")
    b.add_argument("--cutouts", default=None, help="Copy-Paste용 RGBA PNG 폴더")
    b.add_argument("--led-tag", default="_led")
    b.add_argument("--flatfield", default=None, help="기본값: Config.flatfield_path")
    b.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("calib-set", help="Hailo INT8 보정 세트 추출")
    c.add_argument("--tiles", required=True, help="build 출력의 images 폴더")
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=1024)
    c.add_argument("--led-ratio", type=float, default=0.3)
    c.add_argument("--led-tag", default="_led")
    c.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "calibrate":
        {"horizon": calibrate_horizon, "flatfield": calibrate_flatfield,
         "specular": calibrate_specular}[args.what](Config(), args)
    elif args.cmd == "dedup":
        run_dedup(args)
    elif args.cmd == "build":
        run_build(args)
    else:
        run_calib_set(args)


if __name__ == "__main__":
    main()
