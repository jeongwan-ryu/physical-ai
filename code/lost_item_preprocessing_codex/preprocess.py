"""학습·추론에서 함께 사용하는 분실물 검출 전처리.

읽는 순서: Config → Preprocessor.run → quality/correct/tiles → 좌표 변환 함수.
입력 영상은 uint8(0~255), HWC(높이·너비·색상), BGR(OpenCV 색상 순서)이다.
모델 입력은 float32(0~1), NCHW(타일 수·색상·높이·너비), RGB이다.
xyxy 박스는 [왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y]를 뜻한다.
"""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import argparse
import json
import cv2
import numpy as np



@dataclass(frozen=True)
class Config:
    # frozen=True: 생성한 설정을 실수로 바꾸지 못하게 한다.
    # 아래 수치는 HOME-01 초기값이다. 실제 카메라로 측정한 뒤 조정한다.
    # 원본 720행에서 240행을 제외하면 바닥 ROI 높이는 480행이 된다.
    roi_top: int = 240
    # 원본 픽셀 크기를 유지한다. 축소하면 작은 반지·귀걸이가 사라질 수 있다.
    tile_width: int = 672
    tile_height: int = 480
    # 타일 경계에 걸친 물체가 다른 타일에서는 온전히 보이도록 겹친다.
    overlap: int = 64
    # CLAHE의 대비 증폭 제한과 구역 개수. grid는 픽셀 크기가 아니다.
    clip_limit: float = 2.0
    grid: tuple = (8, 3)
    # 어두운 가장자리를 과하게 증폭하면 노이즈도 커지므로 보정 배율 제한.
    gain_cap: float = 2.5
    # 현재 선명도가 최근 중앙값의 60%보다 낮으면 블러로 판단한다.
    blur_ratio: float = 0.6
    # 한 색상 채널이라도 250 이상이면 클리핑 후보로 센다.
    clip_threshold: int = 250
    max_clip_fraction: float = 0.05


def to_linear(image):
    # sRGB 값은 실제 빛의 양에 비례하지 않는다. 조명 배율을 곱하기 전에
    # 전달함수를 역변환해 선형 공간으로 옮긴다. 어두운 구간은 별도 식 사용.
    x = image.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    # 선형 보정 결과를 영상 저장용 sRGB로 되돌린다.
    # clip은 범위를 제한하고 rint는 가까운 정수로 반올림한다.
    x = np.clip(x, 0, 1)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.rint(y * 255).astype(np.uint8)


# Shared CPU lookup tables avoid per-pixel power operations on Raspberry Pi.
# LUT(조회표): 가능한 입력값의 결과를 한 번 계산해 배열 인덱스로 꺼낸다.
# 역변환은 65536단계로 근사하여 CPU의 반복 거듭제곱 계산을 줄인다.
SRGB_TO_LINEAR = to_linear(np.arange(256, dtype=np.uint8))
LINEAR_TO_SRGB = to_srgb(np.arange(65536, dtype=np.float32) / 65535)


def flatfield_correct(image, gain):
    # gain이 (H,W,1)이면 NumPy broadcasting으로 B/G/R 세 채널에 함께 곱한다.
    linear = SRGB_TO_LINEAR[image] * gain
    indices = np.rint(np.clip(linear, 0, 1) * 65535).astype(np.uint16)
    return LINEAR_TO_SRGB[indices]


def read_image(path):
    # 파일 바이트 읽기와 영상 해독을 분리해 Windows 한글 경로도 처리한다.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    # 확장자에 맞춰 PNG/JPEG 등으로 인코딩한 뒤 파일에 기록한다.
    ok, data = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def starts(length, size, overlap):
    """한 축에서 타일 시작 위치 계산. 예: (1280,672,64) → [0,608]."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require tile size > overlap >= 0")
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, size - overlap))
    if result[-1] != length - size:
        # 마지막 타일을 끝에 붙여 오른쪽/아래쪽 영역이 누락되지 않도록 한다.
        result.append(length - size)
    return result


@dataclass
class Tile:
    # x/y는 원본이 아니라 잘라낸 ROI 안에서의 타일 시작 좌표다.
    # valid 크기는 실제 영상 영역으로, 나머지 패딩은 검출 대상이 아니다.
    image: np.ndarray
    x: int
    y: int
    valid_width: int
    valid_height: int


class Preprocessor:
    """Use this exact module for training and inference; no automatic resizing.

    Optional tnr must implement apply(roi_bgr) and reset(). Supply calibrated
    gain/mask in cropped ROI coordinates. One instance per ordered video stream.
    """
    def __init__(self, config=Config(), gain=None, mask=None, tnr=None):
        self.config = config
        if config.roi_top < 0 or config.gain_cap < 1:
            raise ValueError("Invalid ROI or gain cap")
        starts(1, config.tile_width, config.overlap)
        starts(1, config.tile_height, config.overlap)
        self.gain = None if gain is None else np.asarray(gain, dtype=np.float32)
        if self.gain is not None:
            if self.gain.ndim == 2:
                # (H,W) → (H,W,1): 세 색상 채널에 동일 배율을 적용할 준비.
                self.gain = self.gain[..., None]
            if not np.isfinite(self.gain).all() or (self.gain <= 0).any():
                raise ValueError("Gain map must be finite and positive")
            self.gain = np.clip(self.gain, 0, config.gain_cap)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.tnr = tnr
        self.history = deque(maxlen=30)
        # deque는 31번째 값을 넣으면 가장 오래된 값을 자동으로 제거한다.
        self.last_mode = None
        self.clahe = cv2.createCLAHE(config.clip_limit, config.grid)

    def roi(self, frame):
        # 영상 자체를 리사이즈하지 않고 배열 슬라이싱으로 위쪽만 제외한다.
        # 보정 맵·반사 마스크는 이 ROI와 같은 좌표계/해상도여야 한다.
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected uint8 HWC BGR image")
        if self.config.roi_top >= frame.shape[0]:
            raise ValueError("ROI is outside image")
        roi = frame[self.config.roi_top:]
        h, w = roi.shape[:2]
        if self.gain is not None and self.gain.shape not in ((h, w, 1), (h, w, 3)):
            raise ValueError("Gain map shape must match ROI")
        if self.mask is not None and self.mask.shape != (h, w):
            raise ValueError("Reflection mask shape must match ROI")
        return roi

    def reset(self):
        # 새 영상이나 LED 상태 변경 시 이전 장면의 통계를 섞지 않는다.
        self.history.clear()
        self.last_mode = None
        if self.tnr is not None:
            self.tnr.reset()

    def quality(self, frame, led_on=False):
        # 대비 보정 전에 측정해야 CLAHE로 늘어난 엣지를 선명도로 착각하지 않는다.
        roi = self.roi(frame)
        if self.last_mode != led_on:
            self.reset()
            self.last_mode = led_on
        floor = roi[min(120, roi.shape[0] // 4):]
        gray = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (max(1, gray.shape[1] // 2), max(1, gray.shape[0] // 2)))
        # 품질 측정에만 축소 영상을 쓴다. 모델에 넣는 영상은 축소하지 않는다.
        # Laplacian은 밝기의 급격한 변화에 반응한다. 분산은 선명도 대용 지표이며
        # 바닥 질감·노이즈의 영향도 받으므로 절대적인 블러 측정값은 아니다.
        variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        reference = float(np.median(self.history)) if len(self.history) >= 10 else variance
        # 첫 10프레임은 기준 수집 기간. 상대 블러 판정은 그 이후 시작한다.
        # Include every frame so prolonged scene/texture changes can recover.
        self.history.append(variance)
        valid = np.ones(roi.shape[:2], bool) if self.mask is None or not led_on else ~self.mask
        # ~mask는 반사 영역의 반대다. 마스크 밖 픽셀만 분모/분자에 사용한다.
        # max(axis=2)는 각 픽셀의 B/G/R 중 최대값이며 채널 하나만 포화돼도 센다.
        fraction = float(np.mean(roi.max(axis=2)[valid] >= self.config.clip_threshold)) if valid.any() else 0.0
        reason = "overexposed" if led_on and fraction > self.config.max_clip_fraction else (
            "blur" if variance < reference * self.config.blur_ratio else None)
        return {"ok": reason is None, "reason": reason, "laplacian_variance": variance,
                "reference": reference, "clip_fraction": fraction,
                "clipping_measurable": bool(valid.any())}

    def correct(self, frame, led_on=False):
        # copy로 원본 영상과 메모리를 분리해 원본을 보존한다.
        roi = self.roi(frame).copy()
        if led_on:
            if self.tnr is not None:
                roi = self.tnr.apply(roi)
            if self.gain is not None:
                roi = flatfield_correct(roi, self.gain)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # LAB의 L(밝기)만 CLAHE 처리해 색 단서 훼손을 줄인다.
        # 전체 ROI에서 먼저 처리해야 겹친 타일 사이의 밝기 기준이 같다.
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def tiles(self, roi):
        # 모델은 일정한 입력 크기가 필요하다. 작은 영상은 축소/확대 대신
        # 아래/오른쪽을 114 회색으로 채우고 실제 영상 크기를 따로 기록한다.
        c = self.config
        h, w = roi.shape[:2]
        tiles = []
        for y in starts(h, c.tile_height, c.overlap):
            for x in starts(w, c.tile_width, c.overlap):
                piece = roi[y:y + c.tile_height, x:x + c.tile_width]
                ph, pw = piece.shape[:2]
                padded = np.full((c.tile_height, c.tile_width, 3), 114, np.uint8)
                padded[:ph, :pw] = piece
                tiles.append(Tile(padded, x, y, pw, ph))
        batch = np.stack([t.image for t in tiles])[..., ::-1].transpose(0, 3, 1, 2)
        # stack: (N,H,W,3), ::-1: BGR→RGB, transpose: NHWC→NCHW.
        # 축 순서 변경 후에는 메모리가 연속하지 않을 수 있어 contiguous로 복사.
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        batch *= np.float32(1 / 255)
        return batch, tiles

    def run(self, frame, led_on=False, gate=True, probe_frame=False):
        """품질 판정 → ROI 보정 → 타일/정규화. 실패하면 batch는 None.

        gate=False는 학습 등에서 품질 선별을 생략할 때 사용한다.
        probe_frame은 LED를 끄고 주변광을 확인한 프레임이므로 추론에서 제외한다.
        """
        if probe_frame:
            return None, [], {"ok": False, "reason": "probe_frame"}
        metrics = self.quality(frame, led_on) if gate else {"ok": True, "reason": None}
        if not metrics["ok"]:
            return None, [], metrics
        batch, tiles = self.tiles(self.correct(frame, led_on))
        return batch, tiles, metrics


def tile_labels(boxes, tile, roi_top=240, min_visible=0.6, min_size=4):
    """Input Nx5 [class,x1,y1,x2,y2] ORIGINAL pixel coordinates.

    Returns YOLO Nx5 [class,cx,cy,w,h] normalized to padded tile. Tiny or
    heavily clipped boxes are returned separately as local ignore xyxy regions.
    They MUST be used by an ignore-aware loss, not silently treated as background.
    """
    labels, ignored = [], []
    th, tw = tile.image.shape[:2]
    for cls, x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 5):
        # 원본→ROI는 y에서 roi_top 빼기, ROI→타일은 x/y 시작 위치 빼기.
        # max/min으로 타일 밖 박스 부분을 잘라 실제 보이는 교집합만 남긴다.
        original_area = (x2 - x1) * (y2 - y1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid label box")
        a, b = max(0, x1 - tile.x), max(0, y1 - roi_top - tile.y)
        d, e = min(tile.valid_width, x2 - tile.x), min(tile.valid_height, y2 - roi_top - tile.y)
        if d <= a or e <= b:
            continue
        if (d-a)*(e-b)/original_area < min_visible or min(d-a, e-b) < min_size:
            # 잘린 면적 비율과 최소 변 길이를 확인한다. ignore는 배경 라벨이
            # 아니라 학습 loss 계산에서 제외해야 하는 영역이다.
            ignored.append([a, b, d, e])
        else:
            # YOLO 형식: 중심 좌표·너비·높이를 패딩 포함 타일 크기로 나눔.
            labels.append([cls, (a+d)/2/tw, (b+e)/2/th, (d-a)/tw, (e-b)/th])
    return np.asarray(labels, np.float32).reshape(-1, 5), np.asarray(ignored, np.float32).reshape(-1, 4)


def merge_detections(detections, tiles, roi_top=240, mask=None, led_on=False,
                     thresholds=None, negative_classes=(), iou_threshold=0.5, min_size=4):
    """Decoded Nx6 [x1,y1,x2,y2,confidence,class], tile pixel coordinates.
    Class-aware NMS; output full-frame xyxy coordinates. No model decoding here.
    """
    if len(detections) != len(tiles):
        raise ValueError("One detection array required per tile")
    result = []
    thresholds = thresholds or {}
    for rows, tile in zip(detections, tiles):
        # 먼저 모든 타일의 검출을 ROI 좌표로 모은다.
        for x1, y1, x2, y2, confidence, cls in np.asarray(rows).reshape(-1, 6):
            if not np.isfinite([x1,y1,x2,y2,confidence,cls]).all():
                continue
            cls = int(cls)
            if cls in negative_classes:
                continue
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if not (0 <= cx < tile.valid_width and 0 <= cy < tile.valid_height):
                # 회색 패딩에서 생긴 검출은 버린다.
                continue
            x1, x2 = np.clip([x1,x2], 0, tile.valid_width) + tile.x
            y1, y2 = np.clip([y1,y2], 0, tile.valid_height) + tile.y
            if min(x2-x1, y2-y1) < min_size:
                continue
            reflected = led_on and mask is not None and mask[int((y1+y2)/2), int((x1+x2)/2)]
            # 배열 인덱스는 [y,x] 순서다. 반사 영역은 더 높은 확신을 요구한다.
            if confidence < thresholds.get(cls, 0.5) + (0.2 if reflected else 0):
                continue
            result.append([x1, y1+roi_top, x2, y2+roi_top, confidence, cls])
            # ROI에서 원본으로 돌아갈 때 y에 잘라낸 행 수를 다시 더한다.
    kept = []
    for row in sorted(result, key=lambda r: -r[4]):
        # NMS: confidence가 높은 박스부터 유지하고 겹치는 낮은 박스를 제거.
        # 클래스가 다르면 제거하지 않는다. 동일 물체의 타일 중복 검출을 줄인다.
        suppress = False
        for previous in kept:
            if row[5] != previous[5]:
                continue
            a = max(0, min(row[2],previous[2])-max(row[0],previous[0]))
            b = max(0, min(row[3],previous[3])-max(row[1],previous[1]))
            intersection = a*b
            # IoU = 교집합 면적 / 합집합 면적. 겹침 정도를 0~1로 나타낸다.
            union = (row[2]-row[0])*(row[3]-row[1]) + (previous[2]-previous[0])*(previous[3]-previous[1]) - intersection
            if intersection / union > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return np.asarray(kept, np.float32).reshape(-1, 6)


def calibrate_gain(frames, roi_top=240, gain_cap=2.5, sigma=25):
    """Matched LED/exposure/WB matte gray-sheet captures, not scene images."""
    if not frames:
        raise ValueError("At least one flat-field capture required")
    # 색상 채널 평균→여러 촬영의 중앙값으로 우발적인 센서 노이즈를 줄인다.
    field = np.median(np.stack([to_linear(f[roi_top:]).mean(axis=2) for f in frames]), axis=0)
    # 흐리게 하는 대상은 검출 영상이 아니라 보정 맵이다. 조명의 완만한
    # 분포만 남기고 회색 시트의 미세 질감은 보정 배율에 넣지 않는다.
    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), sigma)
    if not np.isfinite(field).all() or np.median(field) <= 1e-6:
        raise ValueError("Flat-field captures are dark or invalid")
    # 기준 밝기/현재 밝기: 어두운 곳은 배율>1, 밝은 곳은 배율<1.
    # 1e-6은 0으로 나누는 것을 막는 아주 작은 값이다.
    return np.clip(np.median(field) / np.maximum(field, 1e-6), 0.1, gain_cap).astype(np.float32)[..., None]


class LedController:
    """Hardware adapter must turn LED off AND wait for valid ambient AE metadata.
    request_probe() only schedules a probe; it does not mark the current frame.
    """
    def __init__(self):
        self.on, self.count, self.last_probe = False, 0, None

    def update(self, exposure_ms, analog_gain, timestamp, probe_frame=False):
        if exposure_ms <= 0 or analog_gain <= 0 or not np.isfinite([exposure_ms, analog_gain, timestamp]).all():
            raise ValueError("Invalid camera metadata")
        index = exposure_ms * analog_gain
        # AE(자동 노출)가 픽셀 밝기를 일정하게 만들 수 있으므로 노출×게인 사용.
        # GPIO를 직접 조작하는 코드가 아니라 필요한 LED 상태를 반환한다.
        if probe_frame:
            self.last_probe = timestamp
            if self.on and index < 60:
                self.on = False
        elif not self.on:
            # 연속 5프레임 조건과 ON/OFF의 서로 다른 임계값(히스테리시스)은
            # 경계 조도에서 LED가 계속 켜졌다 꺼지는 현상을 줄인다.
            self.count = self.count + 1 if index > 100 else 0
            if self.count >= 5:
                self.on, self.count, self.last_probe = True, 0, timestamp
        return self.on

    def request_probe(self, timestamp):
        # 확인 시점만 알려준다. 실제 LED OFF와 AE 안정화는 장비 제어기 담당.
        return self.on and self.last_probe is not None and timestamp - self.last_probe >= 2.0


def main():
    # 터미널 인자를 읽는 CLI 진입점. import해서 사용할 때는 실행되지 않는다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roi-top", type=int, default=240)
    parser.add_argument("--led", action="store_true")
    parser.add_argument("--gain", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--skip-quality-gate", action="store_true")
    parser.add_argument("--calibrate-flatfield", action="store_true")
    parser.add_argument("--threads", type=int, default=2, help="OpenCV CPU threads (default: 2)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    cv2.setNumThreads(args.threads)
    if args.calibrate_flatfield:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, calibrate_gain([read_image(p) for p in args.input], args.roi_top))
        return
    processor = Preprocessor(Config(roi_top=args.roi_top),
                             np.load(args.gain, allow_pickle=False) if args.gain else None,
                             np.load(args.mask, allow_pickle=False) if args.mask else None)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(args.input):
        processor.reset()  # CLI inputs are independent images, not a video stream.
        batch, tiles, metrics = processor.run(read_image(path), args.led, not args.skip_quality_gate)
        prefix = args.output / f"{index:05d}_{path.stem}"
        metadata = {"source": str(path), "led_on": args.led, "quality": metrics,
                    "flatfield_applied": args.led and processor.gain is not None,
                    "color": "RGB", "layout": "NCHW", "tiles": []}
        if batch is not None:
            # NPY는 모델용 숫자 배열, PNG는 사람이 확인하는 영상,
            # JSON은 원본 좌표 복원과 품질 기록에 필요한 부가정보이다.
            np.save(str(prefix) + "_batch.npy", batch)
            for i, tile in enumerate(tiles):
                write_image(str(prefix) + f"_tile{i}.png", tile.image)
                metadata["tiles"].append({"x": tile.x, "y": tile.y + args.roi_top,
                                          "valid_width": tile.valid_width, "valid_height": tile.valid_height})
        Path(str(prefix) + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path.name, metrics)


if __name__ == "__main__":
    main()
