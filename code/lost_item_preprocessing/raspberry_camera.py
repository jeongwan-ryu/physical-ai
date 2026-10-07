"""라즈베리 파이 카메라의 연속 입력을 전처리하는 실행·속도 측정 파일.

역할 구분:
• preprocess.py: 실제 전처리 알고리즘이 들어 있는 공통 모듈.
• 이 파일: 카메라 열기 → 프레임 촬영 → Preprocessor.run() 호출 → 시간 출력.
따라서 실시간 카메라 입력을 처리하지만, 목표 FPS를 보장하는 실시간 제어
시스템은 아니다. YOLO 추론이나 분실물 판정·알림 기능도 아직 연결하지 않았다.

사용 라이브러리:
• Picamera2: CSI 카메라 설정/촬영과 해당 프레임의 노출 메타데이터 읽기.
• OpenCV(cv2): USB 카메라 입력(VideoCapture)과 CPU 스레드 설정.
• NumPy: 보정 NPY 파일 읽기, 측정 시간의 중앙값/p95 계산.
• time.perf_counter(표준): 전처리 함수 호출 전후의 경과시간 측정.
• argparse/pathlib(표준): 실행 옵션과 파일 경로 처리.
• preprocess(직접 작성한 모듈): 공통 Config와 Preprocessor를 가져온다.

실행 예: python3 raspberry_camera.py --source csi --frames 100
USB 예: python3 raspberry_camera.py --source usb --device 0 --frames 100
화면 미리보기나 결과 이미지 저장은 하지 않고 터미널에 품질·속도를 출력한다.
기본은 100프레임 후 종료한다. --frames로 측정 프레임 수를 늘릴 수 있다.
--led는 이미 물리적으로 LED가 켜져 있다는 입력 정보이며 GPIO를 켜지 않는다.
"""
import argparse
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from preprocess import Config, Preprocessor


class CsiCamera:
    """CSI 케이블 카메라용 어댑터. read() → (BGR 영상, 촬영 메타데이터)."""
    def __init__(self, width, height, fps, exposure_ms=None, gain=1.0):
        # USB/PC 실행 때 Picamera2가 없어도 되도록 CSI 생성 시에만 import한다.
        from picamera2 import Picamera2
        self.camera = Picamera2()
        controls = {"FrameRate": fps}
        if exposure_ms is not None:
            # Picamera2 노출 단위는 마이크로초(µs): 4ms = 4000µs.
            # 고정 노출은 AE를 꺼야 적용된다. 화이트밸런스는 여기서 고정하지 않는다.
            controls.update(AeEnable=False, ExposureTime=round(exposure_ms * 1000),
                            AnalogueGain=gain)
        try:
            # Picamera2 RGB888 is B,G,R byte order on little-endian Pi;
            # this matches OpenCV BGR without a second color conversion.
            config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls=controls, buffer_count=4)
            self.camera.configure(config)
            self.camera.start()
        except Exception:
            self.camera.close()
            raise

    def read(self):
        # request는 한 번의 촬영 묶음이다. 영상과 metadata를 별도 캡처하면
        # 서로 다른 프레임의 노출값을 연결할 수 있으므로 같은 묶음에서 꺼낸다.
        request = self.camera.capture_request()
        try:
            # Capture pixels and metadata from the SAME request.
            frame = request.make_array("main").copy()
            # copy로 카메라 버퍼에서 독립시킨 뒤 request를 반환한다.
            metadata = request.get_metadata()
        finally:
            # 예외가 발생해도 반환해야 카메라 버퍼 고갈로 촬영이 멈추지 않는다.
            request.release()
        return frame, {"exposure_ms": metadata.get("ExposureTime", 0) / 1000,
                       "analog_gain": metadata.get("AnalogueGain"),
                       "sensor_timestamp_ns": metadata.get("SensorTimestamp")}

    def close(self):
        try:
            self.camera.stop()
        finally:
            self.camera.close()


class UsbCamera:
    """USB 카메라용 OpenCV 어댑터. 메타데이터는 확인되지 않아 빈 dict 반환."""
    def __init__(self, device, width, height, fps):
        self.camera = cv2.VideoCapture(device)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(f"Cannot open USB camera {device}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # set은 드라이버에 대한 요청이다. 실제 적용 여부는 카메라별로 다르며
        # 메인 루프에서 실제 영상 해상도를 다시 검사한다.

    def read(self):
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("USB frame capture failed")
        # OpenCV USB exposure units are driver-specific; do not invent metadata.
        return frame, {}

    def close(self):
        self.camera.release()


def main():
    # --source로 입력만 바꾸고 공통 Preprocessor는 그대로 사용한다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("csi", "usb"), default="csi")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--roi-top", type=int, default=240)
    parser.add_argument("--led", action="store_true")
    parser.add_argument("--exposure-ms", type=float)
    parser.add_argument("--analog-gain", type=float, default=1.0)
    parser.add_argument("--gain", type=Path)
    parser.add_argument("--mask", type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.frames, args.threads, args.analog_gain) <= 0:
        parser.error("Size, FPS, frames, threads and analog gain must be positive")
    if args.exposure_ms is not None and args.exposure_ms <= 0:
        parser.error("Exposure must be positive")
    if args.source == "usb" and args.exposure_ms is not None:
        parser.error("USB exposure must be configured with the camera driver")
    cv2.setNumThreads(args.threads)
    # CPU 코어를 전처리에 모두 쓰지 않도록 조절한다. 최적값은 실측으로 결정.
    processor = Preprocessor(Config(roi_top=args.roi_top),
                             gain=np.load(args.gain, allow_pickle=False) if args.gain else None,
                             mask=np.load(args.mask, allow_pickle=False) if args.mask else None)
    camera = (CsiCamera(args.width, args.height, args.fps, args.exposure_ms, args.analog_gain)
              if args.source == "csi" else UsbCamera(args.device, args.width, args.height, args.fps))
    durations, accepted = [], 0
    try:
        for index in range(args.frames):
            frame, metadata = camera.read()
            if frame.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Requested {args.width}x{args.height}, received {frame.shape}; recalibrate ROI/maps")
            start = perf_counter()
            # perf_counter는 경과시간 측정용 시계. 초 단위를 ms로 바꿔 출력한다.
            # 촬영 후에 시작하므로 아래 시간은 카메라 대기 시간을 포함하지 않는다.
            batch, tiles, quality = processor.run(frame, led_on=args.led)
            elapsed = (perf_counter() - start) * 1000
            # Accepted-frame statistics include the full preprocessing path.
            if batch is not None:
                # None이면 품질 게이트에서 제외된 프레임이라 추론하면 안 된다.
                durations.append(elapsed)
                accepted += 1
                # Connect inference here; Hailo/TFLite may require NHWC uint8,
                # explicit quantization and per-tile execution instead of NCHW.
            if index % 10 == 0:
                print({"frame": index, "preprocess_ms": round(elapsed, 2),
                       "quality": quality, "camera": metadata,
                       "batch_shape": None if batch is None else batch.shape})
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 또는 오류로 빠져나와도 카메라 자원을 해제한다.
        camera.close()
    # 중앙값은 대표 속도, p95는 성공 프레임의 95%가 이 시간 이하임을 뜻한다.
    # 전처리 시간만 측정하므로 이 값의 역수를 전체 검출 FPS로 보면 안 된다.
    print({"accepted": accepted, "preprocess_ms_median": float(np.median(durations)) if durations else None,
           "preprocess_ms_p95": float(np.percentile(durations, 95)) if durations else None,
           "includes_capture_or_inference": False})


if __name__ == "__main__":
    main()
