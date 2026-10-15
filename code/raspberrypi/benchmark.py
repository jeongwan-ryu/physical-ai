"""
yolo26n 추론 속도 측정 (CPU) - ONNX Runtime / NCNN / OpenVINO / TFLite(LiteRT)

실행:  python benchmark.py --backend onnx              # 웜업 100회 + 측정 300회
       python benchmark.py --backend ncnn --threads 4
       python benchmark.py --backend ncnn --threads 4 --fp16       # NCNN/OpenVINO 만: FP16 연산 (ARM 배포 조건)
       python benchmark.py --backend openvino --warmup 100 --runs 500 --threads 4
       python benchmark.py --backend tflite --warmup 1 --runs 3 --progress 1   # 아주 느린 환경(QEMU 등): 매회 진행 출력

- 모델: models/<포맷>/ (export_models.py, yolotoomnx.py 로 변환). 모두 FP32 가중치, 640x640
- 백엔드마다 바뀌는 건 inference(모델 실행) 뿐, 전처리/후처리/이미지/통계는 공통
- NCNN 은 end2end 헤드를 지원하지 않아 (84, 8400) 출력 + NMS 가 필요 -> NMS 는 postprocess 시간에 포함
- 측정 전에 각 이미지 검출 결과를 ONNX 결과와 비교 (parity) 해서 빠르지만 틀린 백엔드를 걸러냄

- 이미지는 미리 메모리에 읽어 두고 순환 사용 (디스크 I/O 제외)
- 구간별로 따로 측정: preprocess / inference(session.run) / postprocess / total
- 통계: mean, min, p50(중앙값), p90, p95, p99, max(가장 느린 값), std, FPS(p50 기준)
- 필요 패키지: requirements-pi.txt  (torch/ultralytics 불필요 -> 라즈베리파이에서 그대로 실행)
"""
import argparse
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yolotoomnx import CONF, box_iou, postprocess, preprocess

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
STAGES = ("preprocess", "inference", "postprocess", "total")
NMS_IOU = 0.7  # ultralytics 기본값


# ---------------------------------------------------------------- 백엔드
# load_<backend>(threads, fp16) -> (run, decode)
#   run(x: (1,3,640,640) float32) -> 모델 raw 출력
#   decode(raw) -> (1, N, 6) [x1,y1,x2,y2,conf,cls] (letterbox 좌표) -> 공통 postprocess 로 전달
# 기본은 FP32 연산으로 통일 (NCNN/OpenVINO 는 ARM 에서 기본이 FP16 이라 명시적으로 끔)
# --fp16: NCNN/OpenVINO 만 FP16 연산 허용 (ONNX/TFLite 는 FP32 모델 그대로라 해당 없음)
def load_onnx(threads, fp16=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(MODELS / "onnx" / "yolo26n.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return (lambda x: s.run(None, {name: x})[0]), None


def load_openvino(threads, fp16=False):
    import openvino as ov
    cfg = {"INFERENCE_PRECISION_HINT": "f16" if fp16 else "f32"}
    if threads > 0:
        cfg["INFERENCE_NUM_THREADS"] = threads
    m = ov.Core().compile_model(str(MODELS / "openvino" / "yolo26n.xml"), "CPU", cfg)
    req = m.create_infer_request()
    return (lambda x: req.infer({0: x})[0]), None


def load_tflite(threads, fp16=False):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(str(MODELS / "tflite" / "yolo26n.tflite"), num_threads=threads or None)  # 기본: XNNPACK delegate
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def run(x):  # LiteRT export 도 NCHW 입력 그대로
        it.set_tensor(i, x)
        it.invoke()
        return it.get_tensor(o)
    return run, None


def load_ncnn(threads, fp16=False):
    import ncnn
    d = MODELS / "ncnn"
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16  # CPU 가 지원할 때만 실제 적용
    if threads > 0:
        net.opt.num_threads = threads
    net.load_param(str(d / "model.ncnn.param"))
    net.load_model(str(d / "model.ncnn.bin"))

    def run(x):
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(x[0]))
        return np.array(ex.extract("out0")[1])  # (84, 8400): xywh + 80 class score

    def decode(raw):
        p = raw.T
        cls = p[:, 4:].argmax(1)
        conf = p[np.arange(len(p)), 4 + cls]
        p, cls, conf = p[conf > CONF], cls[conf > CONF], conf[conf > CONF]
        xy, wh = p[:, :2], p[:, 2:4]
        keep = cv2.dnn.NMSBoxesBatched(np.c_[xy - wh / 2, wh].tolist(), conf.tolist(), cls.tolist(), CONF, NMS_IOU)
        det = np.c_[xy - wh / 2, xy + wh / 2, conf, cls][np.asarray(keep, dtype=int).reshape(-1)]
        return det.astype(np.float32)[None]
    return run, decode


BACKENDS = {"onnx": load_onnx, "ncnn": load_ncnn, "openvino": load_openvino, "tflite": load_tflite}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="onnx", choices=BACKENDS)
    p.add_argument("--images", default=str(HERE / "images"), help="이미지 폴더 (*.jpg, *.png)")
    p.add_argument("--warmup", type=int, default=100, help="웜업 횟수 (통계에서 제외)")
    p.add_argument("--runs", type=int, default=300, help="측정 횟수")
    p.add_argument("--threads", type=int, default=0, help="intra-op 스레드 수 (0=ORT 기본값)")
    p.add_argument("--fp16", action="store_true", help="FP16 연산 (ncnn, openvino 만)")
    p.add_argument("--csv", help="회차별 측정값 저장 경로 (기본: runs/benchmark_<backend>.csv)")
    p.add_argument("--progress", type=int, default=10, help="N회마다 진행 상황 출력 (0=끔, 1=매회)")
    return p.parse_args()


def load_images(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise SystemExit(f"이미지가 없습니다: {folder}")
    return paths, imgs


def infer_once(run, decode, img):
    """구간별 소요 시간(ms)과 검출 결과를 반환."""
    t0 = time.perf_counter()
    x, r, pad = preprocess(img)
    t1 = time.perf_counter()
    out = run(x)
    t2 = time.perf_counter()
    det = postprocess(decode(out) if decode else out, r, pad, img.shape, conf=CONF)
    t3 = time.perf_counter()
    return ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3), det


def parity(run, decode, paths, imgs):
    """각 이미지 검출 결과를 ONNX Runtime 결과와 비교. 같은 클래스끼리 IoU 최대 매칭."""
    ref_run, _ = load_onnx(0)
    ok = True
    for p, img in zip(paths, imgs):
        a, b = infer_once(ref_run, None, img)[1], infer_once(run, decode, img)[1]
        iou = box_iou(a, b) * (a[:, None, 5] == b[None, :, 5]) if len(a) and len(b) else np.zeros((len(a), 0))
        best = iou.max(1) if iou.size else np.zeros(len(a))
        conf_err = np.abs(a[:, 4] - b[iou.argmax(1), 4]).max() if iou.size else 0.0
        row_ok = len(a) == len(b) and (best > 0.9).all()
        ok &= row_ok
        print(f"parity  : {p.name:<11} onnx={len(a)}개 this={len(b)}개  min IoU={best.min() if len(best) else 1:.3f}  "
              f"max conf_err={conf_err:.3f}  {'PASS' if row_ok else 'FAIL'}")
    return ok


def summarize(ms):
    ms = np.asarray(ms)
    return {
        "mean": ms.mean(), "min": ms.min(),
        "p50": np.percentile(ms, 50), "p90": np.percentile(ms, 90),
        "p95": np.percentile(ms, 95), "p99": np.percentile(ms, 99),
        "max": ms.max(), "std": ms.std(),
    }


def main():
    args = parse_args()
    paths, imgs = load_images(args.images)
    if args.fp16 and args.backend not in ("ncnn", "openvino"):
        raise SystemExit("--fp16 은 ncnn, openvino 만 지원합니다")
    run, decode = BACKENDS[args.backend](args.threads, args.fp16)

    print(f"backend : {args.backend}  threads={args.threads or 'default'}  ({'FP16' if args.fp16 else 'FP32'}, 640x640)")
    print(f"system  : {platform.system()} {platform.machine()}  cpu_count={os.cpu_count()}  "
          f"python={platform.python_version()}")
    print(f"images  : {len(imgs)}개 ({', '.join(p.name for p in paths)}) 순환 사용")
    ok = parity(run, decode, paths, imgs)
    print(f"warmup  : {args.warmup}회 / runs: {args.runs}회\n")

    # 진행 상황: 첫 회, N회마다, 마지막 회에 출력 (docker logs에서도 바로 보이도록 flush)
    total_n, start = args.warmup + args.runs, time.perf_counter()

    def report(done, phase, ms):
        if args.progress <= 0 or not (done == 1 or done % args.progress == 0 or done == total_n):
            return
        elapsed = time.perf_counter() - start
        eta = elapsed / done * (total_n - done)
        print(f"[{phase:<6}] {done:>{len(str(total_n))}}/{total_n}  이번 {ms:9.2f} ms  "
              f"경과 {elapsed:7.1f}s  남은 예상 {eta:7.1f}s", flush=True)

    for i in range(args.warmup):
        report(i + 1, "warmup", infer_once(run, decode, imgs[i % len(imgs)])[0][3])

    records = []
    for i in range(args.runs):
        records.append(infer_once(run, decode, imgs[i % len(imgs)])[0])
        report(args.warmup + i + 1, "run", records[-1][3])
    if args.progress > 0:
        print()
    cols = np.array(records)  # (runs, 4)

    keys = ("mean", "min", "p50", "p90", "p95", "p99", "max", "std")
    print(f"{'stage(ms)':<12}" + "".join(f"{k:>9}" for k in keys))
    print("-" * (12 + 9 * len(keys)))
    for s, stage in enumerate(STAGES):
        st = summarize(cols[:, s])
        print(f"{stage:<12}" + "".join(f"{st[k]:>9.2f}" for k in keys))

    total = summarize(cols[:, 3])
    slowest = int(cols[:, 3].argmax())
    print(f"\nFPS (p50 기준) : {1000 / total['p50']:.1f}")
    print(f"FPS (p95 기준) : {1000 / total['p95']:.1f}")
    print(f"가장 느린 회차 : #{slowest} ({paths[slowest % len(imgs)].name}) total={cols[slowest, 3]:.2f} ms")

    csv = Path(args.csv or HERE / "runs" / f"benchmark_{args.backend}.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv, cols, delimiter=",", fmt="%.4f", header=",".join(f"{s}_ms" for s in STAGES), comments="")
    print(f"회차별 측정값 : {csv}")
    print(f"parity(ONNX 대비) : {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
