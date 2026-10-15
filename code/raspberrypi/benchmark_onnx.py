"""
yolo26n.onnx 추론 속도 측정 (ONNX Runtime, CPU)

실행:  python benchmark_onnx.py                       # 웜업 100회 + 측정 300회
       python benchmark_onnx.py --warmup 100 --runs 500 --threads 4
       python benchmark_onnx.py --warmup 1 --runs 3 --progress 1   # 아주 느린 환경(QEMU 등): 매회 진행 출력

- 이미지는 미리 메모리에 읽어 두고 순환 사용 (디스크 I/O 제외)
- 구간별로 따로 측정: preprocess / inference(session.run) / postprocess / total
- 통계: mean, min, p50(중앙값), p90, p95, p99, max(가장 느린 값), std, FPS(p50 기준)
- 필요 패키지: numpy, opencv-python, onnxruntime  (torch/ultralytics 불필요 -> 라즈베리파이에서 그대로 실행)
"""
import argparse
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yolotoomnx import CONF, postprocess, preprocess

HERE = Path(__file__).resolve().parent
STAGES = ("preprocess", "inference", "postprocess", "total")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(HERE / "yolo26n.onnx"))
    p.add_argument("--images", default=str(HERE), help="이미지 폴더 (*.jpg, *.png)")
    p.add_argument("--warmup", type=int, default=100, help="웜업 횟수 (통계에서 제외)")
    p.add_argument("--runs", type=int, default=300, help="측정 횟수")
    p.add_argument("--threads", type=int, default=0, help="intra-op 스레드 수 (0=ORT 기본값)")
    p.add_argument("--csv", default=str(HERE / "runs" / "benchmark_onnx.csv"), help="회차별 측정값 저장 경로")
    p.add_argument("--progress", type=int, default=10, help="N회마다 진행 상황 출력 (0=끔, 1=매회)")
    return p.parse_args()


def load_images(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise SystemExit(f"이미지가 없습니다: {folder}")
    return paths, imgs


def make_session(model, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])


def infer_once(session, input_name, img):
    """구간별 소요 시간(ms)을 반환."""
    t0 = time.perf_counter()
    x, r, pad = preprocess(img)
    t1 = time.perf_counter()
    out = session.run(None, {input_name: x})[0]
    t2 = time.perf_counter()
    postprocess(out, r, pad, img.shape, conf=CONF)
    t3 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3


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
    session = make_session(args.model, args.threads)
    input_name = session.get_inputs()[0].name

    print(f"model   : {args.model}")
    print(f"input   : {session.get_inputs()[0].shape}  provider={session.get_providers()[0]}  "
          f"ort={ort.__version__}  threads={args.threads or 'default'}")
    print(f"system  : {platform.system()} {platform.machine()}  cpu_count={os.cpu_count()}  "
          f"python={platform.python_version()}")
    print(f"images  : {len(imgs)}개 ({', '.join(p.name for p in paths)}) 순환 사용")
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
        report(i + 1, "warmup", infer_once(session, input_name, imgs[i % len(imgs)])[3])

    records = []
    for i in range(args.runs):
        records.append(infer_once(session, input_name, imgs[i % len(imgs)]))
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

    csv = Path(args.csv)
    csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv, cols, delimiter=",", fmt="%.4f", header=",".join(f"{s}_ms" for s in STAGES), comments="")
    print(f"회차별 측정값 : {csv}")


if __name__ == "__main__":
    main()
