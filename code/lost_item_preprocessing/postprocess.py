"""분실물 프로젝트 후처리: YOLO Results → 원본 좌표 검출 → 지도상 반복 확인.

읽는 순서: PostConfig → LostItemPostprocessor.from_yolo/process → MapConfirmer.
NumPy는 박스 배열/거리 계산, OpenCV는 시각화, dataclasses/deque는 설정과
최근 프레임 저장에 사용한다. Supervision은 선택적 결과 변환에만 필요하다.
카메라 촬영·YOLO 추론·픽셀에서 지도 좌표로의 투영은 호출하는 코드가 담당한다.
"""
from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np
from preprocess import merge_detections


@dataclass(frozen=True)
class PostConfig:
    # 클래스 ID는 학습한 모델에 따라 다르므로 예제 ID를 기본값으로 고정하지 않는다.
    roi_top: int = 240
    thresholds: dict = field(default_factory=dict)  # 예: {반지_ID: 0.35}
    target_classes: tuple | None = None  # None이면 모든 클래스, ()이면 모두 제외
    negative_classes: tuple = ()  # 사료·반사광 등 방해 물체의 실제 클래스 ID
    min_sizes: dict = field(default_factory=dict)  # 클래스별 최소 변 길이(px)
    default_min_size: float = 4.0
    iou_threshold: float = 0.5


def _numpy(value):
    # YOLO Tensor가 GPU에 있다면 CPU로 옮긴다. NumPy 배열 입력도 허용한다.
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class LostItemPostprocessor:
    def __init__(self, config=None, reflection_mask=None):
        self.config = config or PostConfig()
        self.mask = None if reflection_mask is None else np.asarray(reflection_mask, bool)
        if self.mask is not None and self.mask.ndim != 2:
            raise ValueError('Reflection mask must be a 2D ROI mask')
        c = self.config
        if c.roi_top < 0 or not 0 <= c.iou_threshold <= 1 or c.default_min_size <= 0:
            raise ValueError('Invalid postprocessing configuration')
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in c.thresholds.values()):
            raise ValueError('Confidence thresholds must be in [0,1]')
        if any(not np.isfinite(v) or v <= 0 for v in c.min_sizes.values()):
            raise ValueError('Minimum sizes must be positive')

    def from_yolo(self, results, tiles, led_on=False):
        """타일별 Ultralytics Results를 받는다. results 순서는 tiles와 같아야 한다.

        boxes.data를 그대로 쓰지 않고 xyxy/conf/cls를 각각 읽는다. 추적 모드의
        data에는 track ID가 추가될 수 있기 때문이다. supervision 변환과 같은
        역할이지만 NumPy만으로 프로젝트 입력 형식 Nx6을 만든다.
        """
        results = list(results)
        if len(results) != len(tiles):
            raise ValueError('One YOLO result required per tile')
        decoded = []
        for result, tile in zip(results, tiles):
            # Ultralytics Results 좌표는 orig_shape 기준이다. 타일 대신 전체 영상을
            # 추론한 결과를 넣으면 좌표가 어긋나므로 입력 크기를 확인한다.
            if tuple(result.orig_shape) != tuple(tile.image.shape[:2]):
                raise ValueError('YOLO result must refer to the corresponding tile image')
            boxes = result.boxes
            if boxes is None:
                raise ValueError('Detection model boxes required (not classification output)')
            decoded.append(np.column_stack((_numpy(boxes.xyxy), _numpy(boxes.conf),
                                            _numpy(boxes.cls))).astype(np.float32))
        return self.process(decoded, tiles, led_on)

    def process(self, decoded, tiles, led_on=False):
        """타일별 Nx6 [x1,y1,x2,y2,conf,class] → 원본 기준 Nx6.

        TensorRT/ONNX 등에서도 해독된 결과를 같은 형식으로 넣을 수 있다.
        대상/크기 필터를 NMS 전에 적용하고 공통 merge_detections로 좌표 복원,
        반사 필터, 클래스별 NMS를 수행한다. NMS를 중복 실행하지 않는다.
        """
        if len(decoded) != len(tiles):
            raise ValueError('One detection array required per tile')
        if self.mask is not None:
            for t in tiles:
                if t.x < 0 or t.y < 0 or t.x+t.valid_width > self.mask.shape[1] or t.y+t.valid_height > self.mask.shape[0]:
                    raise ValueError('ROI reflection mask does not cover tiles')
        filtered = []
        c = self.config
        for rows, tile in zip(decoded, tiles):
            rows = np.asarray(rows, np.float32).reshape(-1, 6)
            kept = []
            for row in rows:
                if not np.isfinite(row).all() or row[5] != int(row[5]) or not 0 <= row[4] <= 1:
                    continue
                cls = int(row[5])
                if c.target_classes is not None and cls not in c.target_classes:
                    continue
                # 패딩 영역을 잘라낸 후의 실제 박스 크기로 판단한다.
                xs = np.clip(row[[0,2]], 0, tile.valid_width)
                ys = np.clip(row[[1,3]], 0, tile.valid_height)
                if min(xs[1]-xs[0], ys[1]-ys[0]) < c.min_sizes.get(cls, c.default_min_size):
                    continue
                kept.append(row)
            filtered.append(np.asarray(kept, np.float32).reshape(-1, 6))
        return merge_detections(filtered, tiles, roi_top=c.roi_top, mask=self.mask,
                                led_on=led_on, thresholds=c.thresholds,
                                negative_classes=c.negative_classes,
                                iou_threshold=c.iou_threshold, min_size=0)

    @staticmethod
    def to_supervision(rows):
        """필요할 때 Supervision으로 변환. 필터/NMS를 추가 실행하지 않는다."""
        import supervision as sv
        rows = np.asarray(rows, np.float32).reshape(-1, 6)
        return sv.Detections(xyxy=rows[:, :4].copy(), confidence=rows[:, 4].copy(),
                             class_id=rows[:, 5].astype(int))

    @staticmethod
    def annotate(frame, rows, names):
        """원본 이미지의 복사본에 OpenCV로 박스·클래스·신뢰도를 표시한다."""
        out = frame.copy()
        for x1,y1,x2,y2,conf,cls in np.asarray(rows).reshape(-1,6):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 1)
            label = f'{names.get(int(cls), str(int(cls)))} {conf:.2f}'
            # OpenCV 기본 글꼴은 한글을 지원하지 않는다. 영문 라벨을 사용한다.
            cv2.putText(out, label, (x1,max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (0,255,0), 1, cv2.LINE_AA)
        return out


class MapConfirmer:
    """지도 좌표(m) 기준 최근 window프레임 중 need회 관측된 검출을 반환.

    입력 Nx4: [class_id, map_x_m, map_y_m, confidence]. 픽셀 좌표를 넣으면
    안 된다. 카메라 보정/바닥 투영/TF가 먼저 필요하다. 매 카메라 프레임마다
    update 호출: 미검출·품질 탈락·probe 프레임도 빈 배열로 전달한다.
    같은 프레임의 여러 박스는 관측 횟수를 늘리지 않으며 클래스별 일대일
    거리 매칭을 사용한다. 반경 안의 인접 물체는 구분이 어려울 수 있다.
    """
    def __init__(self, radius=0.05, need=3, window=5):
        if not np.isfinite(radius) or radius <= 0 or not 1 <= need <= window:
            raise ValueError('Require radius>0 and 1<=need<=window')
        self.radius, self.need = radius, need
        self.history = deque(maxlen=window-1) if window > 1 else deque(maxlen=0)
        self.last_frame = None

    def reset(self):
        self.history.clear()
        self.last_frame = None

    def update(self, frame_id, observations):
        if self.last_frame is not None and frame_id <= self.last_frame:
            raise ValueError('Frame IDs must increase; call reset() for a new stream')
        if self.last_frame is not None:
            # 호출이 누락된 프레임도 미검출로 처리해 이전 관측이 오래 남지 않게 한다.
            for _ in range(min(frame_id-self.last_frame-1, self.history.maxlen)):
                self.history.append(np.empty((0,4), np.float32))
        current = np.asarray(observations, np.float32).reshape(-1,4)
        if not np.isfinite(current).all() or (current[:,0] != np.floor(current[:,0])).any() or ((current[:,3]<0)|(current[:,3]>1)).any():
            raise ValueError('Invalid map observations')
        # 현재 프레임의 같은 클래스/반경 안 중복을 높은 신뢰도 하나로 줄인다.
        unique = []
        for row in sorted(current, key=lambda r:-r[3]):
            if not any(row[0]==p[0] and np.linalg.norm(row[1:3]-p[1:3]) <= self.radius for p in unique):
                unique.append(row)
        current = np.asarray(unique,np.float32).reshape(-1,4)
        counts = np.ones(len(current),int)
        for previous in self.history:
            # 거리가 가까운 후보부터 매칭하되 프레임마다 양쪽을 한 번씩만 사용.
            candidates = [(float(np.linalg.norm(a[1:3]-b[1:3])),i,j)
                          for i,a in enumerate(current) for j,b in enumerate(previous)
                          if a[0]==b[0] and np.linalg.norm(a[1:3]-b[1:3]) <= self.radius]
            used_current, used_previous = set(), set()
            for distance,i,j in sorted(candidates):
                if i not in used_current and j not in used_previous:
                    counts[i] += 1
                    used_current.add(i)
                    used_previous.add(j)
        self.history.append(current.copy())
        self.last_frame = frame_id
        # 확정 상태인 현재 관측을 반환한다. 알림을 1회만 보내는 기능은 별도로
        # 지도 객체 ID/탐색 세션별 발송 기록에 연결해야 한다.
        return current[counts >= self.need]
