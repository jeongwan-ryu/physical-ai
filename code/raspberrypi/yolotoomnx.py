"""
YOLO26n .pt -> .onnx 변환 + ONNX 추론(bbox/라벨 시각화) + PT/ONNX parity check

실행:  python yolotoomnx.py
출력:
  yolo26n.onnx                 변환된 모델 (라즈베리파이로 복사해서 사용)
  runs/<이미지>_pt.jpg          PyTorch(.pt) 추론 결과
  runs/<이미지>_onnx.jpg        ONNX Runtime 추론 결과

ONNX 추론 부분(letterbox/run_onnx/draw)은 numpy + opencv + onnxruntime 만 사용하므로
라즈베리파이에서는 torch/ultralytics 없이 그대로 가져다 쓸 수 있다.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
PT_PATH = HERE / "yolo26n.pt"
IMAGES = [HERE / "bus.jpg", HERE / "zidane.jpg"]
OUT_DIR = HERE / "runs"
IMGSZ = 640
CONF = 0.25

# parity 허용 오차
RAW_ATOL = 1e-3   # 같은 입력 텐서에 대한 raw 출력 허용 오차: |pt-onnx| <= atol + rtol*|pt|
RAW_RTOL = 1e-4
BOX_ATOL = 1.0    # 최종 bbox 좌표 오차 (픽셀)
CONF_ATOL = 1e-3  # 최종 confidence 오차


# ---------------------------------------------------------------- 1. export
def export_onnx():
    # ultralytics 는 여기서만 import -> 라즈베리파이에서 이 모듈의 ONNX 함수만 가져다 쓸 수 있게
    from ultralytics import YOLO

    # YOLO26은 NMS-free(end2end) 모델. nms=False 를 줘야 end2end 헤드가 그대로 export 되어
    # 출력이 (1, 300, 6) = [x1, y1, x2, y2, conf, cls] 이 된다. (기본값이면 (1, 84, 8400) + 별도 NMS 필요)
    onnx_path = YOLO(str(PT_PATH)).export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True,
                                          dynamic=False, nms=False)
    # export 과정에서 모델 상태가 바뀌지 않도록 parity 비교용 .pt 모델은 새로 로드
    return YOLO(str(PT_PATH)), Path(onnx_path)


# ---------------------------------------------------------------- 2. ONNX 추론 (라즈베리파이용)
def letterbox(img, new_shape=IMGSZ, color=(114, 114, 114)):
    """ultralytics LetterBox(auto=False, center=True)와 동일한 전처리."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (round(w * r), round(h * r))
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr):
    lb, r, pad = letterbox(img_bgr)
    x = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
    return x, r, pad


def postprocess(out, r, pad, orig_shape, conf=CONF):
    """(1,300,6) -> 원본 이미지 좌표의 [N,6] (x1,y1,x2,y2,conf,cls)"""
    det = out[0]
    det = det[det[:, 4] > conf].copy()
    det[:, [0, 2]] = (det[:, [0, 2]] - pad[0]) / r
    det[:, [1, 3]] = (det[:, [1, 3]] - pad[1]) / r
    h, w = orig_shape[:2]
    det[:, [0, 2]] = det[:, [0, 2]].clip(0, w)
    det[:, [1, 3]] = det[:, [1, 3]].clip(0, h)
    return det


def run_onnx(session, img_bgr):
    x, r, pad = preprocess(img_bgr)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    return postprocess(out, r, pad, img_bgr.shape), x, out


def draw(img_bgr, det, names):
    img = img_bgr.copy()
    for x1, y1, x2, y2, conf, cls in det:
        cls = int(cls)
        color = tuple(int(c) for c in np.random.default_rng(cls).integers(0, 255, 3))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = max(p1[1], th + base)
        cv2.rectangle(img, (p1[0], ty - th - base), (p1[0] + tw, ty), color, -1)
        cv2.putText(img, label, (p1[0], ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- 3. parity check
def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def parity_check(pt_model, session, img_bgr, name):
    print(f"\n=== parity: {name} ===")
    ok = True

    # ultralytics(.pt) 추론
    #   nms=False  -> export 와 같은 end2end(one2one) 헤드 사용
    #   rect=False -> ONNX(고정 640x640)와 같은 정사각 letterbox 사용 (기본은 최소 패딩 직사각형)
    r = pt_model.predict(img_bgr, imgsz=IMGSZ, conf=CONF, nms=False, rect=False, verbose=False)[0]

    # (a) raw: 동일한 입력 텐서를 torch 모델과 onnx 모델에 넣어 출력 비교
    import torch
    det_onnx, x, out_onnx = run_onnx(session, img_bgr)
    with torch.no_grad():
        net = pt_model.predictor.model.model  # predict 가 준비해 둔 fused end2end 모델
        out_pt = net(torch.from_numpy(x))
        out_pt = (out_pt[0] if isinstance(out_pt, (list, tuple)) else out_pt).numpy()
    if out_pt.shape != out_onnx.shape:
        print(f"  [raw]   shape mismatch pt={out_pt.shape} onnx={out_onnx.shape}")
        ok = False
    else:
        # 300개 슬롯 중 점수가 낮은 꼬리 부분은 TopK 동점 순서가 바뀔 수 있으므로
        # conf > CONF 인 상위 슬롯만 비교한다 (TopK 결과라 점수 내림차순 정렬되어 있음)
        k = max(int((out_pt[0, :, 4] > CONF).sum()), int((out_onnx[0, :, 4] > CONF).sum()))
        a, b = out_pt[0, :k], out_onnx[0, :k]
        diff = np.abs(a - b)
        raw_ok = np.allclose(b, a, atol=RAW_ATOL, rtol=RAW_RTOL)
        ok &= raw_ok
        print(f"  [raw]   shape={out_onnx.shape}  top{k} max|diff|={diff.max():.2e}  mean|diff|={diff.mean():.2e}  "
              f"-> {'PASS' if raw_ok else 'FAIL'} (atol={RAW_ATOL}, rtol={RAW_RTOL})")

    # (b) 최종 검출: ultralytics(.pt) predict 결과 vs 직접 구현한 ONNX 파이프라인
    det_pt = np.concatenate([r.boxes.xyxy.cpu().numpy(),
                             r.boxes.conf.cpu().numpy()[:, None],
                             r.boxes.cls.cpu().numpy()[:, None]], axis=1)
    print(f"  [det]   pt={len(det_pt)}개  onnx={len(det_onnx)}개")
    if len(det_pt) != len(det_onnx):
        ok = False
    if len(det_pt) and len(det_onnx):
        iou = box_iou(det_pt, det_onnx)
        for i, d in enumerate(det_pt):
            j = int(iou[i].argmax())
            e = det_onnx[j]
            box_err = np.abs(d[:4] - e[:4]).max()
            conf_err = abs(d[4] - e[4])
            same_cls = int(d[5]) == int(e[5])
            row_ok = same_cls and box_err < BOX_ATOL and conf_err < CONF_ATOL
            ok &= row_ok
            print(f"    {pt_model.names[int(d[5])]:>12s}  IoU={iou[i, j]:.4f}  box_err={box_err:.3f}px  "
                  f"conf_err={conf_err:.1e}  {'PASS' if row_ok else 'FAIL'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok, r, det_onnx


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pt_model, onnx_path = export_onnx()
    print(f"\nONNX 저장: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ONNX input :", [(i.name, i.shape) for i in session.get_inputs()])
    print("ONNX output:", [(o.name, o.shape) for o in session.get_outputs()])
    names = pt_model.names

    all_ok = True
    for p in IMAGES:
        img = cv2.imread(str(p))
        ok, r_pt, det_onnx = parity_check(pt_model, session, img, p.name)
        all_ok &= ok
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_pt.jpg"), r_pt.plot())
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_onnx.jpg"), draw(img, det_onnx, names))
        for x1, y1, x2, y2, conf, cls in det_onnx:
            print(f"    onnx: {names[int(cls)]:>12s} {conf:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    print(f"\n결과 이미지: {OUT_DIR}")
    print(f"PARITY {'ALL PASS' if all_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
