"""라즈베리 파이 5 전용 하드웨어 제어: 카메라 · LED · Hailo NPU

================================================================================
이 파일만 따로 둔 이유
    여기 있는 코드는 picamera2, libcamera, Hailo 같은 **라즈베리 파이에만 설치되는 패키지**를
    쓴다. PC에서는 import조차 안 된다. 공용 코드(preprocess.py)와 섞어 두면 PC에서 학습
    데이터를 만들 때 이 패키지들이 없어 오류가 나거나, 코드마다 "Pi인가?" 분기가 생긴다.
    그래서 "Pi에서만 도는 것"은 이 파일 하나에 모았다.

들어 있는 것
    RpiCamera     ① 촬영·ISP 설정 (노출 · 깜빡임 · 노이즈 제거 · 샤프닝 끔 · 초점 고정) + 프레임 읽기
    LedPwm        LED 밝기를 하드웨어 PWM으로 제어
    HailoDetector Hailo NPU에서 YOLO(.hef) 실행 → preprocess.OnboardPipeline의 detector로 사용

설치 (Raspberry Pi OS Bookworm)
    sudo apt install python3-picamera2 hailo-all
    python3 -m venv --system-site-packages .venv    # apt 패키지를 venv에서 쓰려면 이 옵션 필요
================================================================================
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from preprocess import Config, FrameMeta


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ① 카메라 (Picamera2)                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  카메라 내부의 ISP(Image Signal Processor, 영상 처리 전용 칩)가 센서 원본을 사진으로 만들면서
#  노이즈 제거·샤프닝·화이트밸런스 등을 하드웨어로 처리한다. CPU를 전혀 쓰지 않으므로
#  GPU가 없는 라즈베리 파이에서는 "할 수 있는 일은 ISP에 맡기는 것"이 가장 이득이다.
#
#  두 가지 모드
#    개방 공간 (LED 꺼짐) : 자동노출(AE) + 자동 화이트밸런스(AWB) + 조명 깜빡임 보정
#    가구 밑   (LED 켜짐) : 노출 4ms · 게인 · 화이트밸런스를 고정
#                          → 주행 중 번짐이 적고, 플랫필드 보정 맵이 정확히 맞는다

class RpiCamera:
    def __init__(self, cfg: Config):
        from picamera2 import Picamera2          # Pi에서만 있는 패키지라 여기서 import
        from libcamera import controls

        self.cfg = cfg
        self._controls = controls
        self.picam = Picamera2(cfg.rpi.camera_index)
        period_us = int(1_000_000 / cfg.rpi.fps)  # 10fps → 프레임 간격 100,000µs
        config = self.picam.create_video_configuration(
            # "RGB888"이라는 이름과 달리 메모리에는 B,G,R 순서로 들어 있다
            # → OpenCV가 쓰는 BGR 배열과 같아서 변환 없이 바로 쓸 수 있다
            main={"size": (cfg.camera.width, cfg.camera.height), "format": "RGB888"},
            buffer_count=4,
            controls={"FrameDurationLimits": (period_us, period_us)},   # 프레임 속도 고정
        )
        self.picam.configure(config)
        # 이 카메라·libcamera 버전이 지원하는 컨트롤 이름 목록 (버전마다 조금씩 다르다)
        self._supported = set(self.picam.camera_controls.keys())
        self.picam.start()
        self._set(self._common_controls())
        self.fixed = None                  # 현재 모드: True(LED 모드) / False(자동노출) / None(아직 없음)
        self._colour_gains = None          # 자동 화이트밸런스가 마지막으로 정한 색 보정값
        self.set_mode(fixed_exposure=False)

    # ---- 설정 ----
    def _set(self, ctrl: dict):
        """지원하는 컨트롤만 골라서 적용 (지원 안 하는 이름을 넣으면 오류가 나므로)."""
        ok = {k: v for k, v in ctrl.items() if k in self._supported}
        if ok:
            self.picam.set_controls(ok)

    def _common_controls(self):
        """두 모드에 공통인 ISP 설정."""
        c, rpi = self._controls, self.cfg.rpi
        # Sharpness=0: 기본 샤프닝(1.0)은 경계를 강조해서 타일 줄눈·고양이 털이 작은 물체처럼 보인다.
        #              ISP 안에서 일어나는 일이라 나중에 소프트웨어로 되돌릴 수 없으니 여기서 끈다.
        ctrl = {"Sharpness": rpi.sharpness}
        # NoiseReductionMode: 소파 밑 저조도 노이즈를 ISP가 제거. 너무 세면 4px 귀걸이까지 뭉개지므로
        #                     HighQuality / Fast / Minimal을 실측 비교해서 고른다.
        try:
            ctrl["NoiseReductionMode"] = getattr(c.draft.NoiseReductionModeEnum, rpi.noise_reduction)
        except AttributeError:
            pass
        # 초점 고정: Camera Module 3는 오토포커스라 주행 중 초점을 다시 잡느라 흐려질 수 있다.
        #           LensPosition은 디옵터(1/거리m). 3.0이면 약 33cm에 초점.
        if "AfMode" in self._supported:
            ctrl["AfMode"] = c.AfModeEnum.Manual
            ctrl["LensPosition"] = rpi.lens_position
        return ctrl

    def set_mode(self, fixed_exposure: bool):
        """True: LED 모드(노출·게인·화이트밸런스 고정) / False: 개방 공간(자동노출 + 깜빡임 보정)."""
        if fixed_exposure == self.fixed:
            return
        led = self.cfg.led
        if fixed_exposure:
            ctrl = {"AeEnable": False, "ExposureTime": int(led.fixed_exposure_ms * 1000),   # µs 단위
                    "AnalogueGain": float(led.fixed_gain), "AwbEnable": False}
            if self._colour_gains is not None:
                ctrl["ColourGains"] = self._colour_gains    # 직전 자동 화이트밸런스 값으로 고정
        else:
            ctrl = {"AeEnable": True, "AwbEnable": True}
            # 깜빡임 보정: 노출 시간을 1/120초(8333µs)의 배수로 맞추면 조명이 깜빡여도 매 프레임
            # 같은 양의 빛을 받아서 가로 줄무늬·밝기 출렁임이 사라진다.
            try:
                ctrl["AeFlickerMode"] = self._controls.AeFlickerModeEnum.Manual
                ctrl["AeFlickerPeriod"] = self.cfg.rpi.flicker_period_us
            except AttributeError:
                pass
        self._set(ctrl)
        self.fixed = fixed_exposure

    # ---- 촬영 ----
    def capture(self):
        """프레임 한 장 → (BGR uint8 프레임, FrameMeta, 원본 메타데이터 dict)."""
        req = self.picam.capture_request()
        try:
            frame = req.make_array("main")
            md = req.get_metadata()          # 이 프레임의 실제 노출 시간·게인 등
        finally:
            req.release()                    # 버퍼를 카메라에 돌려줘야 다음 프레임을 받을 수 있다
        if not self.fixed and "ColourGains" in md:
            self._colour_gains = tuple(md["ColourGains"])   # LED 모드로 바뀔 때 고정할 값으로 기억
        ts = md.get("SensorTimestamp", time.monotonic_ns()) / 1e9
        meta = FrameMeta(exposure_ms=md.get("ExposureTime", 0) / 1000.0,
                         gain=md.get("AnalogueGain", 1.0), timestamp=ts)
        return frame, meta, md

    def drop_frames(self, n: int):
        """설정을 바꾼 직후 프레임 n장을 버린다.

        카메라 설정은 몇 프레임 뒤에 적용되고, 롤링셔터 카메라는 위에서 아래로 한 줄씩 찍기 때문에
        찍는 도중에 LED가 바뀌면 위쪽 반만 밝은 프레임이 나온다. 이런 프레임은 버린다.
        """
        for _ in range(n):
            self.picam.capture_request().release()

    def close(self):
        self.picam.stop()
        self.picam.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ LED 밝기 제어 (하드웨어 PWM)                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  PWM: LED를 아주 빠르게 켰다 껐다 하면서 켜진 시간 비율(duty)로 밝기를 조절하는 방식.
#  주파수가 낮으면 카메라가 한 줄씩 찍는 동안 LED가 꺼진 줄이 생겨 가로 줄무늬가 찍힌다.
#  그래서 20kHz 이상으로, 소프트웨어가 아닌 하드웨어 PWM(RP1 칩)으로 만든다.
#
#  사전 설정: /boot/firmware/config.txt 에 `dtoverlay=pwm-2chan` 추가 후 재부팅.
#  리눅스는 /sys/class/pwm/pwmchipN/ 아래 파일에 숫자를 써서 PWM을 제어한다(sysfs).
#  chip 번호는 커널 버전마다 달라서 `ls /sys/class/pwm`으로 확인해 PwmConfig.chip에 넣는다.

class LedPwm:
    def __init__(self, cfg: Config):
        p = cfg.pwm
        self.cfg = p
        self.base = f"/sys/class/pwm/pwmchip{p.chip}"
        self.ch = f"{self.base}/pwm{p.channel}"
        if not os.path.exists(self.ch):              # 채널을 처음 쓰면 export로 활성화
            self._write(f"{self.base}/export", p.channel)
            time.sleep(0.1)
        self.period = int(1e9 / p.freq_hz)           # 한 주기 길이(ns). 20kHz → 50,000ns
        self._write(f"{self.ch}/period", self.period)
        self.duty = p.duty
        self.on = False
        self._write(f"{self.ch}/duty_cycle", 0)
        self._write(f"{self.ch}/enable", 1)

    @staticmethod
    def _write(path, value):
        with open(path, "w") as f:
            f.write(str(value))

    def _apply(self):
        # duty_cycle = 한 주기 중 켜져 있는 시간(ns). 꺼짐이면 0.
        self._write(f"{self.ch}/duty_cycle", int(self.period * self.duty) if self.on else 0)

    def set(self, on: bool):
        self.on = bool(on)
        self._apply()

    def dim(self):
        """과노출 판정을 받으면 밝기를 한 단계(5%) 낮춘다. 최소 10%까지."""
        self.duty = max(self.cfg.duty_min, self.duty - self.cfg.duty_step)
        self._apply()

    def close(self):
        self._write(f"{self.ch}/duty_cycle", 0)
        self._write(f"{self.ch}/enable", 0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Hailo NPU 검출기                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
#  Hailo는 ONNX를 직접 못 돌리고, PC에서 Hailo Dataflow Compiler로 변환한 .hef 파일만 실행한다
#  (변환 방법은 README 참고). HEF 안에 YOLO 후처리(NMS)까지 넣어 컴파일했다고 가정한다.
#  그러면 결과는 "클래스별로 [y1, x1, y2, x2, score] 목록"이고 좌표는 0~1 비율이다.
#  ※ x보다 y가 먼저 오는 순서에 주의.

class HailoDetector:
    """사용법: pipe = OnboardPipeline(cfg, HailoDetector("model.hef"))"""

    def __init__(self, hef_path: str, score_th: float = 0.2):
        from picamera2.devices import Hailo      # picamera2에 들어 있는 Hailo 실행 도우미
        self.hailo = Hailo(hef_path)
        self.in_h, self.in_w = self.hailo.get_input_shape()[:2]
        # 여기서는 낮게(0.2) 거르고, 클래스별 기준은 preprocess.filter_detections에서 적용한다
        self.score_th = score_th

    def __call__(self, tiles):
        """타일 리스트 → 타일별 (N,6) [x1, y1, x2, y2, conf, cls] (타일 픽셀 좌표)."""
        out = []
        for t in tiles:
            h, w = t.shape[:2]
            if (h, w) != (self.in_h, self.in_w):
                raise ValueError(f"타일 {w}x{h} != HEF 입력 {self.in_w}x{self.in_h}")
            # Hailo 입력은 uint8 RGB 그대로 (/255 정규화는 HEF 안에서 함)
            res = self.hailo.run(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
            out.append(self._parse(res, w, h))
        return out

    def _parse(self, res, w, h):
        """Hailo NMS 출력 → (N,6). 0~1 비율 좌표에 타일 크기를 곱해 픽셀 좌표로."""
        dets = []
        for cls, boxes in enumerate(res):              # 리스트 위치 = 클래스 번호
            for b in boxes:
                y1, x1, y2, x2, s = (float(v) for v in b[:5])
                if s >= self.score_th:
                    dets.append([x1 * w, y1 * h, x2 * w, y2 * h, s, cls])
        return np.array(dets, np.float32).reshape(-1, 6)

    def close(self):
        self.hailo.close()
