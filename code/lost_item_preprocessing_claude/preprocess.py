"""분실물 검출 무인이동체 — 전처리 핵심 모듈 (PC · 라즈베리 파이 공용)

================================================================================
이 파일 하나에 "로봇이 프레임 한 장을 받아서 분실물을 확정하기까지"의 모든 계산이
처리 순서대로 들어 있다. 위에서 아래로 읽으면 실제 데이터가 흐르는 순서와 같다.

    [카메라 ISP] 노이즈 제거 · 샤프닝 끔 · 초점 고정        ← rpi_hardware.py 가 설정
         │
         ▼  프레임 (1280×720, BGR)
    ② LED 점등 판단 ............ LedController
    ③ 품질 게이트 .............. QualityGate      (흐린 프레임·과노출 프레임 버리기)
    ④ ROI 크롭 ................. Preprocessor.crop (쓸모없는 위쪽 240행 잘라내기)
    ⑤ 플랫필드 보정 ............ Preprocessor.apply_flatfield (LED 조명 얼룩 펴기)
    ⑥ CLAHE .................... Preprocessor.apply_clahe     (구역별 대비 올리기)
    ⑦ 타일링 ................... split_tiles      (해상도 유지한 채 2조각으로 나누기)
         │
         ▼  [Hailo NPU] YOLO 추론                ← rpi_hardware.HailoDetector
         │
    ⑧ 후처리 ................... merge_tiles → filter_detections → MultiFrameConfirmer
         │
         ▼  "에어팟 확정! 지도 위치 (x, y)"

    위 단계를 한 번에 돌려 주는 것이 맨 아래의 OnboardPipeline.step() 이다.

왜 파일을 이렇게 나눴나? (실행 환경 기준)
    preprocess.py     PC · Pi 공용. numpy + OpenCV만 쓴다.
                      학습 데이터 만들 때(PC)와 로봇이 추론할 때(Pi) **같은 코드**를 써야
                      학습-추론 불일치가 생기지 않는다. 그래서 공용 코드는 한 곳에만 둔다.
    rpi_hardware.py   Pi 전용. picamera2 · libcamera · Hailo 같은 Pi에만 있는 패키지를 쓴다.
    offline_tools.py  PC 전용. 증강 · 데이터셋 생성 · 캘리브레이션 같은 오프라인 작업.
    run_demo.py       실행 진입점. 셀프테스트 · 녹화 영상 데모 · Pi 실시간 실행.

용어
    ROI       Region Of Interest. 관심 영역. 여기서는 "화면에서 바닥이 보이는 아래쪽 1280×480".
    선형 공간  카메라 픽셀값(0~255)은 사람 눈에 맞춰 휘어 있다(감마). 이걸 펴서 "실제 빛의 양"에
              비례하게 만든 값(0~1)이 선형 값이다. 빛을 곱하고 더하는 계산은 선형 공간에서 해야 맞다.
    타일      큰 영상을 모델 입력 크기(672×480)로 자른 조각.
    conf      모델이 낸 확신도(0~1).
================================================================================
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 0. 설정값                                                                    ║
# ║    숫자를 코드 곳곳에 흩어 놓지 않고 여기 한곳에 모았다. 현장 실측 후 여기만    ║
# ║    고치면 된다. 모든 값은 가상 현장 HOME-01 기준 "초기값"이다.                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 모델이 구분하는 클래스 목록. 리스트 위치(0, 1, 2, ...)가 곧 클래스 번호다.
# 6~11번 "방해 물체"는 학습에는 넣지만(모델이 '이건 에어팟이 아니라 사료야'를 배우도록)
# 최종 결과에서는 버린다. 배경으로 두는 것보다 오검출을 훨씬 확실히 줄인다.
CLASS_NAMES = [
    "airpods_bud", "airpods_case", "ring", "earring", "key", "remote",   # 0~5: 찾는 물건
    "kibble", "pill", "bottle_cap", "hair_tie", "plug", "led_glare",    # 6~11: 헷갈리는 물건
]
TARGET_CLASSES = frozenset(range(0, 6))
NEGATIVE_CLASSES = frozenset(range(6, 12))
GLARE_CLASS = CLASS_NAMES.index("led_glare")    # = 11. 증강에서 가짜 반사점 라벨로 쓴다

_HFOV_DEG = 102.0   # Camera Module 3 Wide의 수평 화각(도)


@dataclass
class CameraConfig:
    """카메라 렌즈 특성과 로봇에 달린 자세. 체커보드 캘리브레이션 값으로 바꿔야 정확하다."""
    width: int = 1280
    height: int = 720
    # fx, fy: 초점거리(픽셀 단위). "1m 앞의 1m짜리 물체가 화면에서 몇 px로 보이나"와 같다.
    # 화각으로 근사: 화면 절반(640px)이 화각 절반(51°)에 해당 → fx = 640 / tan(51°) ≈ 518px
    fx: float = 640.0 / math.tan(math.radians(_HFOV_DEG / 2))
    fy: float = 640.0 / math.tan(math.radians(_HFOV_DEG / 2))
    cx: float = 640.0                          # 광축이 화면을 뚫는 점(보통 화면 중앙)
    cy: float = 360.0
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)    # 렌즈 왜곡 계수 k1, k2, p1, p2, k3
    height_m: float = 0.05                     # 바닥에서 렌즈 중심까지 높이 (5cm)
    pitch_deg: float = 15.0                    # 카메라를 아래로 숙인 각도
    offset_x_m: float = 0.0                    # 로봇 기준점에서 카메라까지 앞쪽 거리


@dataclass
class RpiCameraConfig:
    """라즈베리 파이 카메라(Picamera2) 설정. rpi_hardware.RpiCamera가 사용."""
    camera_index: int = 0
    fps: float = 10.0
    sharpness: float = 0.0             # ISP 샤프닝 끔. 기본값 1.0은 타일 줄눈·고양이 털을 강조한다
    noise_reduction: str = "HighQuality"   # HighQuality / Fast / Minimal / Off 중 실측으로 고른다
    lens_position: float = 3.0         # 초점 고정. 단위 디옵터 = 1/거리(m). 3.0 → 약 33cm에 초점
    flicker_period_us: int = 8333      # 한국 전원 60Hz → 조명은 초당 120번 깜빡 → 1/120초 = 8333µs
    settle_frames_exposure: int = 3    # 노출 모드를 바꾼 뒤 버릴 프레임 수 (설정이 늦게 적용됨)
    settle_frames_led: int = 1         # LED를 켜고 끈 뒤 버릴 프레임 수 (반쪽만 밝은 프레임 방지)


@dataclass
class PwmConfig:
    """LED 밝기 조절용 하드웨어 PWM. `ls /sys/class/pwm`으로 chip 번호를 확인할 것."""
    chip: int = 0
    channel: int = 0                   # dtoverlay=pwm-2chan 기준: 채널0 = GPIO18, 채널1 = GPIO19
    freq_hz: int = 20000               # 20kHz 이상이어야 롤링셔터 카메라에 가로 줄무늬가 안 생긴다
    duty: float = 0.6                  # 처음 밝기 (60%)
    duty_min: float = 0.1
    duty_step: float = 0.05            # 과노출일 때 한 번에 낮추는 폭


@dataclass
class PreprocConfig:
    """④~⑦ 전처리 설정. 학습(PC)과 추론(Pi)이 똑같이 쓴다."""
    roi_top: int = 240                 # 위에서 240행을 버린다 → 1280×720 이 1280×480 이 된다
    tile_w: int = 672                  # 타일 크기. 672와 480은 둘 다 32의 배수(YOLO 요구사항)
    tile_h: int = 480
    tile_overlap: int = 64             # 타일끼리 최소 64px 겹치게 → 경계에 걸린 작은 물체도 온전히 한 타일에 들어감
    clahe_clip: float = 2.0            # CLAHE가 대비를 얼마나 세게 올릴지 (클수록 강함, 노이즈도 커짐)
    clahe_grid: tuple = (8, 3)         # 화면을 가로 8 × 세로 3 칸으로 나눠 칸마다 대비 조정 (칸 ≈ 160×160px)
    ff_gain_min: float = 0.5           # 플랫필드 게인 범위: 어두운 곳을 최대 2.5배까지만 밝힌다
    ff_gain_max: float = 2.5           #   (더 키우면 노이즈만 커진다)


@dataclass
class QualityConfig:
    """③ 품질 게이트 설정."""
    blur_ratio: float = 0.6            # 선명도가 최근 평소값의 60% 미만이면 "흐린 프레임"
    history: int = 30                  # 평소값 = 최근 30프레임 선명도의 중앙값
    min_history: int = 10              # 기록이 10개 모이기 전에는 판정하지 않음
    eval_row_start: int = 120          # ROI 안에서 120행 아래(가까운 바닥)만 보고 선명도 측정
    downscale: float = 0.5             # 선명도는 절반 크기로 줄여서 계산 (빠르고 노이즈에 덜 민감)
    clip_level: int = 250              # 250 이상이면 "하얗게 날아간(포화) 픽셀"로 본다
    clip_ratio_max: float = 0.05       # LED 모드에서 포화 픽셀이 5%를 넘으면 과노출 프레임


@dataclass
class LedConfig:
    """② LED 점등 판단 설정."""
    on_index: float = 100.0            # 노출(ms) × 게인 이 100을 넘으면 "어둡다"
    on_frames: int = 5                 # 5프레임 연속 어두우면 켠다 (한두 프레임 그림자에 흔들리지 않게)
    probe_interval_s: float = 2.0      # LED를 켠 동안 2초마다 1프레임 꺼서 주변이 밝아졌는지 확인
    probe_off_mean: float = 70.0       # 그 확인 프레임의 평균 밝기(0~255)가 70 이상이면 LED를 끈다
    fixed_exposure_ms: float = 4.0     # LED 모드 노출 4ms (짧을수록 주행 중 번짐이 적다)
    fixed_gain: float = 4.0            # LED 모드 아날로그 게인 (4 이하로 노이즈 억제)
    target_mean: tuple = (140, 160)    # LED 모드에서 바닥 중앙 밝기 목표 (PWM 조정 참고값)


@dataclass
class PostConfig:
    """⑧ 후처리 설정."""
    nms_iomin: float = 0.6             # 타일 병합 때 "같은 물체"로 볼 겹침 비율
    conf_th: dict = field(default_factory=lambda: {
        # 클래스별 확신도 기준. 작은 물건(이어버드·반지·귀걸이)은 원래 확신도가 낮게 나와서 기준도 낮춘다
        0: 0.35, 1: 0.5, 2: 0.35, 3: 0.35, 4: 0.5, 5: 0.5,
    })
    default_conf_th: float = 0.5
    specular_conf_bonus: float = 0.2   # LED 반사점 자리에서는 확신도를 0.2 더 요구 (반사광을 에어팟으로 착각 방지)
    min_box_px: int = 4                # 4px보다 작은 박스는 버린다
    confirm_radius_m: float = 0.05     # 지도에서 5cm 안이면 "같은 물체"
    confirm_need: int = 3              # 최근 5프레임 중 3프레임 이상 보이면 확정
    confirm_window: int = 5


@dataclass
class Config:
    """모든 설정을 한 묶음으로. 사용법: cfg = Config(); cfg.pre.clahe_clip = 3.0"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    rpi: RpiCameraConfig = field(default_factory=RpiCameraConfig)
    pwm: PwmConfig = field(default_factory=PwmConfig)
    pre: PreprocConfig = field(default_factory=PreprocConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    led: LedConfig = field(default_factory=LedConfig)
    post: PostConfig = field(default_factory=PostConfig)
    flatfield_path: str = "calib/flatfield_led.npy"      # offline_tools.py calibrate flatfield 결과
    specular_mask_path: str = "calib/specular_mask.npy"  # offline_tools.py calibrate specular 결과

    def roi_shape(self):
        """ROI 영상의 (높이, 너비) = (720-240, 1280) = (480, 1280)."""
        return (self.camera.height - self.pre.roi_top, self.camera.width)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 1. 기본 도구: 이미지 파일 입출력, 감마 ↔ 선형 변환                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread 대체. 실패하면 None.

    왜 필요한가: Windows에서 cv2.imread는 경로에 한글(예: C:\\Users\\작업용)이 있으면
    오류도 없이 None을 돌려준다. 파일을 바이트로 읽은 뒤(np.fromfile) 메모리에서
    디코딩(cv2.imdecode)하면 경로 문제를 피할 수 있다.
    """
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img) -> None:
    """cv2.imwrite 대체. 한글 경로에서도 동작하고, 실패하면 조용히 넘어가지 않고 예외를 낸다."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError(f"이미지 인코딩 실패: {path}")
    buf.tofile(path)


# 감마 ↔ 선형 변환
# ----------------------------------------------------------------------------
# 카메라는 실제 빛의 양 L(0~1)을 그대로 저장하지 않고, 어두운 부분을 더 촘촘히 표현하려고
#     픽셀값 = 255 × L^(1/2.2)
# 로 휘어서 저장한다(감마 인코딩). 그래서 "빛을 2배로" 같은 물리 계산을 하려면 먼저
#     L = (픽셀값/255)^2.2
# 로 펴야 한다(선형화). 매번 거듭제곱을 계산하면 느리므로 미리 표(LUT)를 만들어 둔다.
#
#   _TO_LIN : 픽셀값 0~255 → 선형 값 (256칸짜리 표)
#   _TO_SRGB: 선형 값을 65536단계로 쪼갠 칸 → 픽셀값 (65536칸짜리 표)
#             16bit로 촘촘하게 쪼갠 이유: 소파 밑(2 lux)처럼 아주 어두운 값도 뭉개지지 않게.
_GAMMA = 2.2
_LIN_LEVELS = 65536
_TO_LIN = ((np.arange(256) / 255.0) ** _GAMMA).astype(np.float32)
_TO_SRGB = np.round((np.arange(_LIN_LEVELS) / (_LIN_LEVELS - 1)) ** (1 / _GAMMA) * 255).astype(np.uint8)


def to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 픽셀(0~255) → float32 선형 값(0~1). 표를 인덱스로 찾는 것이라 빠르다."""
    return _TO_LIN[img_u8]


def to_srgb(lin: np.ndarray) -> np.ndarray:
    """float 선형 값(0~1) → uint8 픽셀. 1을 넘는 값은 255로 잘린다(= 하얗게 날아감)."""
    q = np.clip(lin, 0.0, 1.0) * (_LIN_LEVELS - 1) + 0.5     # 0~65535 칸 번호로 변환 (+0.5는 반올림)
    return _TO_SRGB[q.astype(np.uint16)]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 2. 카메라 기하: 화면 속 픽셀이 바닥의 어디인가                                ║
# ║    용도: ① 검출된 물건의 지도 좌표 계산  ② Copy-Paste 증강에서 물체 크기 계산    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#   옆에서 본 그림 (카메라는 바닥에서 5cm 높이, 15° 아래를 봄)
#
#        카메라 ●──────────────── 광축 (15° 아래로 기울어짐)
#           5cm│  ╲  ╲
#              │    ╲    ╲  ← 화면 아래쪽 픽셀일수록 가까운 바닥을 본다
#   ───────────┴──────●───────●────── 바닥
#                    가까움   멀다
#
#   픽셀 하나 = 카메라에서 나가는 광선 하나. 그 광선이 바닥(높이 0)과 만나는 점을 구하면
#   "그 픽셀에 찍힌 물건이 로봇 앞 몇 m, 왼쪽 몇 m에 있는지" 알 수 있다.
#   지평선보다 위를 향하는 광선은 바닥과 만나지 않는다(무효).

class CameraGeometry:
    def __init__(self, cam: CameraConfig):
        self.cam = cam
        # K: 카메라 행렬. 3D 방향 ↔ 픽셀 좌표 변환에 쓰는 표준 형식
        self.K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
        self.D = np.array(cam.dist, np.float64)
        th = math.radians(cam.pitch_deg)
        self._sin, self._cos = math.sin(th), math.cos(th)

    def undistort_norm(self, pts_px) -> np.ndarray:
        """픽셀 좌표 (N,2) → 렌즈 왜곡을 걷어 낸 '정규화 좌표' (N,2).

        정규화 좌표 (x, y): 광축에서 오른쪽으로 x, 아래로 y만큼 떨어진 방향 (거리 1 기준).
        영상 전체를 펴지(Undistort) 않고 필요한 점만 펴는 이유: 영상을 펴면 보간 때문에
        4~10px짜리 작은 물체가 흐려지고, CPU 시간도 든다.
        """
        p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        if len(p) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(p, self.K, self.D).reshape(-1, 2)

    def pixel_to_floor(self, pts_px):
        """원본 프레임 픽셀 (N,2) → 로봇 기준 바닥 좌표 (N,2) [앞 m, 왼쪽 m], 유효 여부 (N,).

        바닥에 놓인 물체는 박스 '하단 중앙'(바닥에 닿은 점)을 넣어야 위치가 맞는다.
        박스 중심을 넣으면 물체 높이만큼 더 멀리 있는 것으로 계산된다.
        """
        n = self.undistort_norm(pts_px)
        x, y = n[:, 0], n[:, 1]
        # 광선 방향을 로봇 좌표로 바꾸면 (카메라가 15° 숙여져 있으므로 회전 적용):
        #   앞쪽 성분 = cos - y·sin,  왼쪽 성분 = -x,  아래쪽 성분 = y·cos + sin
        down = y * self._cos + self._sin
        valid = down > 1e-6                          # 아래쪽 성분이 있어야 바닥과 만난다
        # 높이 h에서 출발해 아래로 내려가 바닥에 닿을 때까지 광선을 늘리는 배율 t = h / 아래쪽 성분
        t = self.cam.height_m / np.where(valid, down, 1.0)
        fwd = t * (self._cos - y * self._sin) + self.cam.offset_x_m
        left = t * (-x)
        out = np.stack([fwd, left], axis=1)
        out[~valid] = np.nan
        return out, valid

    def horizon_row(self) -> float:
        """지평선이 화면 몇 번째 행인지 (화면 중앙 열 기준 근사). 이보다 위는 바닥이 아니다."""
        return self.cam.cy - self.cam.fy * (self._sin / self._cos)

    def object_size_px(self, rows, cols, size_m: float) -> np.ndarray:
        """그 픽셀 위치 바닥에 놓인 '가장 긴 변이 size_m인 물체'가 화면에서 몇 px로 보이는지.

        원리: 화면 크기(px) ≈ 초점거리(px) × 실제 크기(m) / 거리(m)
        예) 반지 2cm, 1m 거리 → 518 × 0.02 / 1 ≈ 10px
        """
        pts = np.stack([np.asarray(cols, np.float64), np.asarray(rows, np.float64)], axis=1)
        floor, valid = self.pixel_to_floor(pts)
        fwd = floor[:, 0] - self.cam.offset_x_m
        rng = np.sqrt(fwd ** 2 + floor[:, 1] ** 2 + self.cam.height_m ** 2)   # 렌즈~물체 직선거리
        size = self.cam.fx * size_m / rng
        size[~valid] = np.nan
        return size


def robot_to_map(pts_xy: np.ndarray, pose) -> np.ndarray:
    """로봇 기준 좌표 (N,2) → 지도 좌표 (N,2). pose = 로봇의 지도상 (x, y, 바라보는 각도 rad).

    회전 후 이동하는 2D 좌표 변환이다. ROS 2에서는 tf2로 base_link → map 변환을 쓰면 된다.
    """
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = pts_xy[:, 0], pts_xy[:, 1]
    return np.stack([x + c * px - s * py, y + s * px + c * py], axis=1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ② LED 점등 판단                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  "화면 평균 밝기가 어두우면 LED를 켠다"는 동작하지 않는다.
#  카메라 자동노출(AE)이 어두우면 노출을 늘리고 게인을 올려서 화면 평균을 늘 중간쯤으로
#  맞춰 버리기 때문이다. 대신 "카메라가 얼마나 애쓰고 있나"를 본다:
#
#      밝기 지수 = 노출 시간(ms) × 아날로그 게인
#      밝은 거실 180 lux → 노출 8ms × 게인 1.5 = 12     (여유 있음)
#      소파 밑 2 lux     → 노출 100ms × 게인 8 = 800    (한계까지 애씀) → LED 켜자
#
#  문제: LED를 켜면 노출을 4ms로 고정하므로 지수가 더는 변하지 않는다 → 밖으로 나왔는지 모름.
#  해결: 2초마다 딱 1프레임 LED를 끄고 찍어서(확인용 프레임), 그 사진이 충분히 밝으면 LED를 끈다.
#
#  상태 변화:
#       OFF ──(5프레임 연속 어두움)──▶ ON ──(2초 경과)──▶ PROBE(LED 끄고 1장 촬영)
#        ▲                                                     │
#        └────────(확인 사진이 밝음)────────────────────────────┤
#                                    ON ◀───(여전히 어두움)─────┘
#
#  명령은 "다음 프레임부터" 적용된다고 가정한다(카메라·LED 하드웨어 특성).

@dataclass
class LedCommand:
    led_on: bool          # 다음 프레임에 LED를 켤지
    fixed_exposure: bool  # True: 노출 4ms·게인 고정(LED 모드) / False: 자동노출(개방 공간)
    use_frame: bool       # 이번 프레임을 검출에 써도 되는지 (확인용 프레임은 False)
    frame_led_on: bool    # 이번 프레임이 LED 켠 상태로 찍혔는지 → 전처리 방식 선택에 사용


class LedController:
    OFF, ON, PROBE = "off", "on", "probe"

    def __init__(self, cfg: LedConfig):
        self.cfg = cfg
        self.state = self.OFF
        self._count = 0                        # 연속으로 어두웠던 프레임 수
        self._last_probe = float("-inf")       # 마지막으로 확인 프레임을 찍은 시각

    @property
    def needs_roi_mean(self) -> bool:
        """이번 프레임이 확인용 프레임이라 평균 밝기를 계산해 줘야 하는지."""
        return self.state == self.PROBE

    def step(self, exposure_ms: float, gain: float, now: float, roi_mean: float = 0.0) -> LedCommand:
        c = self.cfg

        # [OFF] LED 꺼짐: 어두운 프레임이 연속 몇 장인지 센다
        if self.state == self.OFF:
            self._count = self._count + 1 if exposure_ms * gain > c.on_index else 0
            if self._count >= c.on_frames:
                self.state, self._count, self._last_probe = self.ON, 0, now
                return LedCommand(led_on=True, fixed_exposure=True, use_frame=True, frame_led_on=False)
            return LedCommand(led_on=False, fixed_exposure=False, use_frame=True, frame_led_on=False)

        # [ON] LED 켜짐: 2초가 지났으면 다음 프레임은 LED를 끄고 찍도록 예약
        if self.state == self.ON:
            if now - self._last_probe >= c.probe_interval_s:
                self.state = self.PROBE
                return LedCommand(led_on=False, fixed_exposure=True, use_frame=True, frame_led_on=True)
            return LedCommand(led_on=True, fixed_exposure=True, use_frame=True, frame_led_on=True)

        # [PROBE] 지금 프레임은 LED를 끄고 (노출은 고정한 채) 찍은 확인용 사진이다
        #   노출이 같으므로 "이 사진의 밝기 = 순수한 주변광의 밝기"로 비교할 수 있다
        self._last_probe = now
        if roi_mean >= c.probe_off_mean:       # 주변이 충분히 밝다 → LED 끄고 자동노출로 복귀
            self.state = self.OFF
            return LedCommand(led_on=False, fixed_exposure=False, use_frame=False, frame_led_on=False)
        self.state = self.ON                   # 아직 어둡다 → 다시 LED 켬
        return LedCommand(led_on=True, fixed_exposure=True, use_frame=False, frame_led_on=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ③ 품질 게이트: 쓸모없는 프레임은 계산하기 전에 버린다                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  (a) 흐린 프레임: 주행 중 흔들림으로 10px 반지가 2px만 번져도 모양을 잃는다.
#      선명도 = Laplacian(2차 미분, 경계에서 크게 반응) 값의 분산. 흐리면 경계가 뭉개져 작아진다.
#      고정 기준값을 쓰지 않는 이유: 무늬 없는 타일 바닥은 원래 선명도가 낮아서 멀쩡한 프레임까지
#      버리게 된다. 그래서 "최근 30프레임의 평소값보다 확 떨어졌나"로 판단한다.
#
#  (b) 과노출 프레임: LED가 너무 세서 하얗게 날아간 픽셀이 5% 넘으면 버리고 LED를 줄이라고 알린다.
#      단, 유광 타일의 LED 반사점(늘 같은 자리)은 어차피 날아가므로 계산에서 뺀다.
#
#  CLAHE 전에 검사하는 이유: CLAHE가 대비를 올리면 흐린 사진도 선명해 보여서 통과해 버린다.

@dataclass
class QualityResult:
    ok: bool
    reason: str = None      # "blur"(흐림) | "overexposed"(과노출) | None(통과)
    sharpness: float = 0.0
    clip_ratio: float = 0.0


class QualityGate:
    def __init__(self, cfg: QualityConfig, roi_top: int, specular_mask=None):
        self.cfg = cfg
        self.roi_top = roi_top
        self.mask = specular_mask
        # 반사점 자리를 0, 나머지를 255로 표시한 마스크. 포화 픽셀 계산에서 반사점을 빼는 데 쓴다.
        # numpy 대신 OpenCV 함수로 계산하는 이유: Pi CPU에서 numpy의 max(axis=2)가 10배쯤 느리다.
        self._valid = None if specular_mask is None else np.where(specular_mask, 0, 255).astype(np.uint8)
        self._n_valid = None if specular_mask is None else max(int(np.count_nonzero(~specular_mask)), 1)
        self._hist = deque(maxlen=cfg.history)   # 최근 선명도 기록 (꽉 차면 오래된 것부터 빠짐)
        self._mode = None

    def reset(self):
        self._hist.clear()

    def check(self, frame: np.ndarray, led_on: bool) -> QualityResult:
        c = self.cfg
        # LED를 켜고 끄면 영상의 선명도 수준 자체가 달라지므로 기록을 새로 시작한다
        if led_on != self._mode:
            self.reset()
            self._mode = led_on

        # (a) 선명도: ROI 아래쪽(가까운 바닥)만, 흑백으로, 절반 크기로 줄여서 계산
        roi = frame[self.roi_top:]
        g = cv2.cvtColor(roi[c.eval_row_start:], cv2.COLOR_BGR2GRAY)
        if c.downscale != 1.0:
            g = cv2.resize(g, None, fx=c.downscale, fy=c.downscale, interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())

        ok, reason = True, None
        if len(self._hist) >= c.min_history and sharp < c.blur_ratio * float(np.median(self._hist)):
            ok, reason = False, "blur"
        self._hist.append(sharp)   # 흐린 프레임도 기록에 넣는다 → 바닥이 바뀌어 계속 낮아도 기준이 따라 내려감

        # (b) 포화 비율: 세 채널(B,G,R) 중 가장 밝은 값이 250 이상인 픽셀의 비율
        b, g, r = cv2.split(roi)
        clip = cv2.compare(cv2.max(cv2.max(b, g), r), c.clip_level, cv2.CMP_GE)   # 포화=255, 아니면 0
        if self._valid is not None:
            clip = cv2.bitwise_and(clip, self._valid)       # 반사점 자리는 0으로 지움
            clip_ratio = cv2.countNonZero(clip) / self._n_valid
        else:
            clip_ratio = cv2.countNonZero(clip) / clip.size
        if ok and led_on and clip_ratio > c.clip_ratio_max:
            ok, reason = False, "overexposed"               # → 호출한 쪽에서 LED 밝기를 낮춘다
        return QualityResult(ok, reason, sharp, clip_ratio)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ④⑤⑥ 결정적 전처리: ROI 크롭 → 플랫필드 → CLAHE                               ║
# ║    "결정적" = 같은 입력이면 항상 같은 출력 (랜덤 없음).                         ║
# ║    학습 데이터 생성과 로봇 추론이 반드시 이 클래스를 같이 써야 한다.             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_flatfield(path, roi_shape_hw, gain_min, gain_max):
    """플랫필드 게인 맵 파일(.npy) 읽기. (H,W) 또는 (H,W,3) → (H,W,1|3). 파일이 없으면 None."""
    if not path or not os.path.exists(path):
        return None
    g = np.load(path).astype(np.float32)
    if g.ndim == 2:
        g = g[..., None]               # (480,1280) → (480,1280,1): 세 채널에 같은 게인을 곱하기 위해
    if g.shape[:2] != tuple(roi_shape_hw):
        raise ValueError(f"flatfield shape {g.shape[:2]} != ROI {tuple(roi_shape_hw)}")
    return np.clip(g, gain_min, gain_max)


def load_mask(path, roi_shape_hw):
    """LED 반사점 마스크 파일(.npy) 읽기. (H,W) bool, True = 반사점 자리. 파일이 없으면 None."""
    if not path or not os.path.exists(path):
        return None
    m = np.load(path).astype(bool)
    if m.shape != tuple(roi_shape_hw):
        raise ValueError(f"mask shape {m.shape} != ROI {tuple(roi_shape_hw)}")
    return m


class Preprocessor:
    """프레임 → 전처리된 ROI 영상. 사용법: pre = Preprocessor(cfg.pre, flatfield); img = pre(frame, led_on)"""

    def __init__(self, cfg: PreprocConfig, flatfield=None):
        self.cfg = cfg
        self.flatfield = flatfield
        # 플랫필드 빠른 계산 준비
        # ------------------------------------------------------------------
        # 원래 식: 결과 = sRGB( 선형(픽셀) × 게인 )   ← 매 프레임 거듭제곱 2번, Pi에서 느림
        # 감마 2.2 모델에서는 (게인 × L)^(1/2.2) = 게인^(1/2.2) × L^(1/2.2) 이므로
        #         결과 = 픽셀 × 게인^(1/2.2)          ← 곱셈 1번이면 끝 (오차 ±1 이내)
        # 그래서 게인^(1/2.2)를 한 번만 미리 계산해 두고, 프레임마다 cv2.multiply만 한다.
        self._ff_srgb = None
        if flatfield is not None:
            g = flatfield if flatfield.shape[2] == 3 else np.repeat(flatfield, 3, axis=2)
            self._ff_srgb = np.ascontiguousarray(g ** (1 / _GAMMA), dtype=np.float32)
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=tuple(cfg.clahe_grid))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """④ ROI 크롭: 위쪽 240행(소파 밑면·벽)을 버린다.

        카메라가 5cm 높이에서 15° 아래를 보므로 지평선은 약 221행이다. 그 위에는 바닥이 없으니
        분실물이 있을 수 없다. 잘라내면 이후 모든 계산량이 1/3 줄어든다.
        frame[240:]은 복사가 아니라 '보는 범위만 바꾼 것'(view)이라 비용이 0이다.
        """
        return frame[self.cfg.roi_top:]

    def apply_flatfield(self, roi: np.ndarray) -> np.ndarray:
        """⑤ 플랫필드 보정: LED 조명의 얼룩(가운데만 밝고 가장자리 어두움)을 평평하게 편다.

        원리: 찍힌 밝기 = 조명 세기 × 바닥 반사율. 조명 세기 분포를 미리 재 두었다가
        (무광 회색 시트 촬영 → offline_tools.py calibrate flatfield) 그 역수를 곱하면
        "조명이 고르게 비췄을 때의 사진"이 된다.
        로봇 카메라와 LED 위치가 고정이고 바닥은 평면이라 조명 분포가 늘 같아서 가능한 방법이다.
        255를 넘으면 255로 잘린다(cv2.multiply가 자동 처리).
        """
        if self._ff_srgb is None:
            return roi
        return cv2.multiply(roi, self._ff_srgb, dtype=cv2.CV_8U)

    def apply_clahe(self, roi: np.ndarray) -> np.ndarray:
        """⑥ CLAHE (Contrast Limited Adaptive Histogram Equalization): 구역별로 대비를 올린다.

        흰 에어팟이 밝은 회색 타일 위에 있으면 밝기 차이가 작아서 잘 안 보인다. CLAHE는 화면을
        8×3 칸으로 나눠 칸마다 밝기 분포를 넓혀 준다. 화면 전체를 한 번에 하는 일반 평활화와 달리
        그림자 진 구역과 밝은 구역을 각각 알맞게 조정한다. clipLimit는 과하게 올려서 노이즈까지
        키우는 것을 막는 상한이다.

        LAB 색공간으로 바꿔 L(밝기) 채널에만 거는 이유: RGB 각각에 걸면 색이 틀어진다.
        흰 에어팟 vs 갈색 사료 vs 금 귀걸이를 구분하는 '색' 단서는 그대로 지켜야 한다.
        """
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def __call__(self, frame: np.ndarray, led_on: bool) -> np.ndarray:
        """원본 프레임 → 전처리된 ROI 영상 (480×1280). 순서가 중요하다:

        크롭 먼저   : 이후 계산량을 줄이고, 쓸모없는 소파 밑면이 CLAHE 통계를 왜곡하지 않게
        플랫필드 다음: 곱셈(선형) 보정이라 비선형인 CLAHE보다 먼저 해야 원리가 맞는다
        CLAHE 마지막: 타일로 자르기 '전에' 전체 화면에 걸어야 두 타일의 밝기 기준이 같다
        """
        roi = self.crop(frame)
        if led_on:                              # 플랫필드 맵은 LED 조명용이라 LED 모드에서만
            roi = self.apply_flatfield(roi)
        return self.apply_clahe(roi)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ⑦ 타일링: 해상도를 줄이지 않고 모델 입력 크기로 자르기                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  1280×480 ROI를 YOLO 입력(672×480) 하나로 줄이면 반지 10px → 5px로 사라진다.
#  대신 원본 해상도 그대로 두 조각으로 자른다. 두 조각은 64px 겹친다.
#
#      x: 0                    608   672                  1280
#         ├─────── 타일 0 ───────┼─────┤
#                                ├────────── 타일 1 ──────────┤
#                                 겹침 64px
#
#  겹치는 이유: 경계에 걸친 물체가 양쪽에서 반쪽씩 잘리면 못 찾는다. 겹침 폭(64px)보다 작은
#  물체는 적어도 한 타일 안에는 온전히 들어간다.

def tile_offsets(width: int, tile_w: int, min_overlap: int):
    """타일 시작 x 좌표 목록. 겹침이 min_overlap 이상이 되도록 균등 배치. 1280/672/64 → [0, 608]."""
    if width <= tile_w:
        return [0]
    step = tile_w - min_overlap
    n = math.ceil((width - tile_w) / step) + 1                   # 필요한 타일 개수
    return [round(i * (width - tile_w) / (n - 1)) for i in range(n)]   # 처음은 0, 마지막은 오른쪽 끝에 맞춤


def split_tiles(img: np.ndarray, cfg: PreprocConfig):
    """전처리된 ROI 영상 → (타일 리스트[uint8 BGR], 각 타일 시작 x 리스트)."""
    h, w = img.shape[:2]
    if h != cfg.tile_h:
        raise ValueError(f"ROI 높이 {h} != tile_h {cfg.tile_h}. roi_top 또는 tile_h를 맞출 것")
    xs = tile_offsets(w, cfg.tile_w, cfg.tile_overlap)
    # ascontiguousarray: 잘라낸 조각을 메모리상 연속된 배열로 복사 (모델·OpenCV가 요구)
    return [np.ascontiguousarray(img[:, x:x + cfg.tile_w]) for x in xs], xs


def tiles_to_hailo(tiles) -> np.ndarray:
    """타일 리스트 → (N, H, W, 3) uint8 RGB.

    Hailo 모델(HEF)은 0~255 정수 RGB를 그대로 받는다. /255 정규화는 HEF 안에 들어 있으므로
    여기서 또 나누면 입력이 거의 검게 들어간다. OpenCV는 BGR 순서라 [..., ::-1]로 뒤집는다.
    """
    return np.ascontiguousarray(np.stack(tiles)[..., ::-1])


# 학습용 박스 좌표 변환 (offline_tools.py의 데이터셋 생성에서 사용)
# 박스 형식: (N,5) [클래스, x1, y1, x2, y2] — 왼쪽 위(x1,y1), 오른쪽 아래(x2,y2) 픽셀 좌표

def boxes_frame_to_roi(boxes: np.ndarray, roi_top: int, roi_h: int, min_visible=0.6) -> np.ndarray:
    """원본 프레임 좌표 → ROI 좌표 (y에서 240을 뺀다). ROI 밖으로 40% 넘게 잘린 박스는 버린다."""
    return _clip_boxes(boxes, 0, roi_top, None, roi_h, min_visible)


def boxes_roi_to_tile(boxes: np.ndarray, x0: int, tile_w: int, tile_h: int, min_visible=0.6) -> np.ndarray:
    """ROI 좌표 → 타일 좌표 (x에서 타일 시작점을 뺀다). 타일 밖으로 40% 넘게 잘린 박스는 버린다."""
    return _clip_boxes(boxes, x0, 0, tile_w, tile_h, min_visible)


def _clip_boxes(boxes, x0, y0, w, h, min_visible):
    """박스를 (x0, y0)만큼 옮기고 (0~w, 0~h) 범위로 자른 뒤, 원래 면적의 min_visible 이상 남은 것만 남긴다."""
    if len(boxes) == 0:
        return boxes.reshape(0, 5)
    b = boxes.astype(np.float32).copy()
    b[:, [1, 3]] -= x0
    b[:, [2, 4]] -= y0
    area = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])               # 자르기 전 면적
    xmax = np.inf if w is None else w
    c = b.copy()
    c[:, [1, 3]] = np.clip(c[:, [1, 3]], 0, xmax)
    c[:, [2, 4]] = np.clip(c[:, [2, 4]], 0, h)
    vis = (c[:, 3] - c[:, 1]) * (c[:, 4] - c[:, 2])                # 자른 뒤 남은 면적 (밖이면 음수)
    keep = (area > 0) & (vis / np.maximum(area, 1e-6) >= min_visible)
    return c[keep]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ⑧ 후처리: 타일 병합 → 필터 → 여러 프레임에 걸쳐 확인                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  검출 배열 형식: (N,6) float32 [x1, y1, x2, y2, conf, cls]

EMPTY = np.zeros((0, 6), np.float32)   # "검출 없음"을 나타내는 빈 배열 (쓸 때는 .copy())


def _suppress(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """중복 박스 제거(NMS, Non-Maximum Suppression). 남길 박스 번호를 돌려준다.

    방법: 확신도 높은 박스부터 하나씩 남기고, 그 박스와 많이 겹치는 나머지는 지운다.

    겹침 기준으로 보통 쓰는 IoU(교집합/합집합) 대신 IoMin(교집합/작은 박스 면적)을 쓴다.
    타일 경계에서 반쪽만 잡힌 박스와 온전한 박스는
        IoU   = 반쪽 / 온전 = 0.5 정도 → 같은 물체인데 못 지움
        IoMin = 반쪽 / 반쪽 = 1.0     → 확실히 지움
    """
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]          # 확신도 내림차순 번호
    keep = []
    while order.size:
        i = order[0]                        # 남은 것 중 확신도 1등은 남긴다
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)  # 교집합 너비
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)  # 교집합 높이
        iomin = iw * ih / (np.minimum(areas[i], areas[rest]) + 1e-9)
        order = rest[iomin <= thr]          # 많이 겹치는 것은 지우고 나머지로 반복
    return np.array(keep, dtype=int)


def merge_tiles(dets_per_tile, xs, iomin_thr: float) -> np.ndarray:
    """타일별 검출 결과 → ROI 좌표로 합치고, 겹침 영역의 중복을 클래스별로 지운다."""
    parts = []
    for d, x0 in zip(dets_per_tile, xs):
        if d is None or len(d) == 0:
            continue
        d = np.asarray(d, np.float32).copy()
        d[:, [0, 2]] += x0                  # 타일 좌표 → ROI 좌표 (타일 시작 x만큼 오른쪽으로)
        parts.append(d)
    if not parts:
        return EMPTY.copy()
    dets = np.concatenate(parts)
    keep = []
    for c in np.unique(dets[:, 5]):         # 클래스가 다르면 겹쳐도 지우지 않는다
        idx = np.where(dets[:, 5] == c)[0]
        keep.extend(idx[_suppress(dets[idx, :4], dets[idx, 4], iomin_thr)])
    return dets[np.sort(keep)]


def filter_detections(dets: np.ndarray, cfg: PostConfig, specular_mask=None) -> np.ndarray:
    """검출 결과 걸러내기 (ROI 좌표 기준).

    1) 방해 물체 클래스(사료·알약·반사광 등)는 버린다
    2) 클래스별 확신도 기준 미만은 버린다
       단, 박스 중심이 LED 반사점 자리면 기준을 0.2 올린다 (반사광을 에어팟으로 착각 방지)
    3) 4px보다 작은 박스는 버린다
    """
    if len(dets) == 0:
        return EMPTY.copy()
    cls = dets[:, 5].astype(int)
    th = np.array([cfg.conf_th.get(c, cfg.default_conf_th) for c in cls], np.float32)
    if specular_mask is not None:
        h, w = specular_mask.shape
        cx = np.clip(((dets[:, 0] + dets[:, 2]) / 2).astype(int), 0, w - 1)
        cy = np.clip(((dets[:, 1] + dets[:, 3]) / 2).astype(int), 0, h - 1)
        th = th + cfg.specular_conf_bonus * specular_mask[cy, cx]   # 마스크 안이면 True(=1) × 0.2
    size = np.minimum(dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1])
    keep = (~np.isin(cls, list(NEGATIVE_CLASSES))) & (dets[:, 4] >= th) & (size >= cfg.min_box_px)
    return dets[keep]


@dataclass
class Track:
    """지도 위 한 지점에서 관찰된 물체 기록."""
    cls: int
    pos: np.ndarray                  # 지도 좌표 (x, y), 관찰할 때마다 평균으로 갱신
    n_obs: int = 0                   # 지금까지 관찰된 횟수
    best_conf: float = 0.0
    hits: deque = field(default_factory=deque)   # 관찰된 프레임 번호들 (최근 window 프레임만 유지)
    confirmed: bool = False


class MultiFrameConfirmer:
    """여러 프레임에서 같은 자리에 반복해서 보여야 "진짜 물건"으로 확정한다.

    규칙: 지도 좌표 5cm 안에서, 최근 5프레임 중 3프레임 이상 같은 클래스가 검출되면 확정.

    왜 효과가 있나:
      - 진짜 물건: 로봇이 움직여도 지도 좌표는 그대로 → 같은 Track에 계속 쌓인다
      - LED 반사점: 화면 속 같은 자리에 붙어 다니므로 로봇이 움직이면 지도 좌표가 계속 바뀐다
                     → 매번 다른 Track이 되어 3번을 못 채운다
      - 한두 프레임 깜빡 나온 오검출: 3번을 못 채운다
    """

    def __init__(self, radius_m: float, need: int, window: int):
        self.radius = radius_m
        self.need = need
        self.window = window
        self.tracks = []

    def reset(self):
        self.tracks.clear()

    def update(self, frame_idx: int, observations):
        """observations: [(클래스, 지도좌표 np.array([x,y]), conf), ...] → 이번에 새로 확정된 Track 리스트."""
        # 1) 관찰마다 가장 가까운 같은 클래스 Track을 찾아 붙이거나, 없으면 새 Track을 만든다
        for cls, xy, conf in observations:
            best, best_d = None, self.radius
            for t in self.tracks:
                if t.cls != cls:
                    continue
                d = float(np.linalg.norm(t.pos - xy))
                if d <= best_d:
                    best, best_d = t, d
            if best is None:
                best = Track(cls, np.asarray(xy, np.float64).copy())
                self.tracks.append(best)
            else:
                best.pos = (best.pos * best.n_obs + xy) / (best.n_obs + 1)   # 누적 평균 위치
            best.n_obs += 1
            best.best_conf = max(best.best_conf, float(conf))
            if not best.hits or best.hits[-1] != frame_idx:                  # 한 프레임은 한 번만 셈
                best.hits.append(frame_idx)

        # 2) 오래된 기록을 지우고, 기준을 채운 Track을 확정한다
        newly, alive = [], []
        for t in self.tracks:
            while t.hits and t.hits[0] <= frame_idx - self.window:
                t.hits.popleft()
            if not t.confirmed and len(t.hits) >= self.need:
                t.confirmed = True
                newly.append(t)
            if t.confirmed or t.hits:          # 확정된 Track은 계속 남겨 둔다 → 같은 물건을 또 알리지 않음
                alive.append(t)
        self.tracks = alive
        return newly


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 전체 조립: 프레임 한 장 → (LED 명령, 검출, 확정) 을 돌려주는 파이프라인          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_UNSET = object()   # "인자를 안 줬음"을 None과 구분하기 위한 표식


@dataclass
class FrameMeta:
    """카메라가 프레임과 함께 알려 주는 정보 (Picamera2 메타데이터에서 가져옴)."""
    exposure_ms: float          # 노출 시간 (ms)
    gain: float                 # 아날로그 게인 (배)
    timestamp: float            # 촬영 시각 (초)


@dataclass
class StepResult:
    """OnboardPipeline.step()의 결과."""
    led: LedCommand                              # 다음 프레임의 LED·노출 명령
    quality: QualityResult = None                # 품질 검사 결과 (검사 전에 끝났으면 None)
    processed: np.ndarray = None                 # 전처리된 ROI 영상 (시각화용)
    detections: np.ndarray = field(default_factory=EMPTY.copy)   # (N,6) 원본 프레임 좌표, 필터 통과분
    map_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # 각 검출의 지도 좌표
    confirmed: list = field(default_factory=list)  # 이번 프레임에 새로 확정된 Track (보통 비어 있음)


class OnboardPipeline:
    """로봇 온보드 파이프라인. ROS 2와 무관한 순수 Python이라 PC에서도 녹화 영상으로 돌려 볼 수 있다.

    사용법:
        pipe = OnboardPipeline(Config(), detector)       # detector: 아래 형식의 함수/객체
        res = pipe.step(frame, meta, robot_pose)
        LED·노출 장치에 res.led 적용, res.confirmed가 있으면 알림

    detector 형식: detector(tiles: list[uint8 BGR]) -> list[(N,6) 배열, 타일 좌표 기준]
        로봇에서는 rpi_hardware.HailoDetector, PC 검증에서는 run_demo.UltralyticsDetector.
    """

    def __init__(self, cfg: Config = None, detector=None, flatfield=_UNSET, specular_mask=_UNSET):
        self.cfg = cfg = cfg or Config()
        roi_hw = cfg.roi_shape()
        # 캘리브레이션 파일: 인자로 직접 주지 않으면 설정된 경로에서 읽는다 (없으면 보정 생략)
        if flatfield is _UNSET:
            flatfield = load_flatfield(cfg.flatfield_path, roi_hw, cfg.pre.ff_gain_min, cfg.pre.ff_gain_max)
        if specular_mask is _UNSET:
            specular_mask = load_mask(cfg.specular_mask_path, roi_hw)
        self.spec_mask = specular_mask
        self.geo = CameraGeometry(cfg.camera)
        self.pre = Preprocessor(cfg.pre, flatfield)
        self.gate = QualityGate(cfg.quality, cfg.pre.roi_top, specular_mask)
        self.led = LedController(cfg.led)
        self.confirmer = MultiFrameConfirmer(cfg.post.confirm_radius_m, cfg.post.confirm_need,
                                             cfg.post.confirm_window)
        self.detector = detector
        self.force_led = None       # None: LED 자동 판단 / True·False: 녹화 영상처럼 LED 상태를 이미 알 때 고정
        self.frame_idx = 0

    def step(self, frame: np.ndarray, meta: FrameMeta, robot_pose=(0.0, 0.0, 0.0)) -> StepResult:
        """프레임 한 장 처리. 중간에 쓸모없는 프레임으로 판정되면 거기서 바로 돌아온다."""
        cfg = self.cfg
        self.frame_idx += 1

        # ② LED 판단 ─ 다음 프레임의 LED·노출 명령을 정하고, 이번 프레임을 써도 되는지 본다
        if self.force_led is not None:
            on = bool(self.force_led)
            cmd = LedCommand(on, on, True, on)
        else:
            roi_mean = 0.0
            if self.led.needs_roi_mean:      # 확인용 프레임일 때만 평균 밝기를 계산 (평소엔 계산 낭비)
                roi_mean = float(cv2.mean(cv2.cvtColor(frame[cfg.pre.roi_top:], cv2.COLOR_BGR2GRAY))[0])
            cmd = self.led.step(meta.exposure_ms, meta.gain, meta.timestamp, roi_mean)
        res = StepResult(cmd)
        if not cmd.use_frame:                # 확인용 프레임은 검출에 쓰지 않는다
            return res
        mode = cmd.frame_led_on              # 이 프레임이 LED 켠 상태로 찍혔는지

        # ③ 품질 게이트 ─ 전처리 '전' 원본으로 검사
        res.quality = self.gate.check(frame, mode)
        if not res.quality.ok:
            return res

        # ④⑤⑥ 결정적 전처리 ─ 학습 데이터 만들 때와 똑같은 코드
        img = self.pre(frame, mode)
        res.processed = img
        if self.detector is None:            # 검출기 없이 전처리 결과만 보고 싶을 때
            return res

        # ⑦ 타일링 + 추론 (로봇에서는 Hailo NPU)
        tiles, xs = split_tiles(img, cfg.pre)
        raw = self.detector(tiles)

        # ⑧ 병합 + 필터 (ROI 좌표) → 원본 프레임 좌표로 (y에 240을 다시 더함)
        dets = merge_tiles(raw, xs, cfg.post.nms_iomin)
        dets = filter_detections(dets, cfg.post, self.spec_mask)
        dets[:, [1, 3]] += cfg.pre.roi_top
        res.detections = dets

        # 박스 하단 중앙(바닥에 닿은 점) → 로봇 기준 바닥 좌표 → 지도 좌표
        obs = []
        if len(dets):
            ground = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], axis=1)
            floor, valid = self.geo.pixel_to_floor(ground)
            pts = robot_to_map(np.nan_to_num(floor), robot_pose)
            res.map_points = pts
            obs = [(int(d[5]), p, float(d[4])) for d, p, ok in zip(dets, pts, valid) if ok]

        # 다중 프레임 확정 ─ 검출이 없는 프레임에도 호출해야 오래된 기록이 정리된다
        res.confirmed = self.confirmer.update(self.frame_idx, obs)
        return res
