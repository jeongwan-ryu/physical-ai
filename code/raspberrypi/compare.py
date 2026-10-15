"""
백엔드별 측정 결과(benchmark_<backend>.csv) 비교표 생성

실행:  python compare.py runs/compare_docker_amd64     # 폴더 안 benchmark_*.csv -> summary.md
"""
import sys
from pathlib import Path

import numpy as np

folder = Path(sys.argv[1])
rows = []
for csv in sorted(folder.glob("benchmark_*.csv")):
    d = np.loadtxt(csv, delimiter=",", skiprows=1)  # preprocess, inference, postprocess, total
    inf, tot = d[:, 1], d[:, 3]
    rows.append((csv.stem.removeprefix("benchmark_"), len(d),
                 np.median(inf), np.median(d[:, 2]), np.median(tot), np.percentile(tot, 95), tot.max(),
                 1000 / np.median(tot), 1000 / np.percentile(tot, 95)))
assert rows, f"benchmark_*.csv 없음: {folder}"
rows.sort(key=lambda r: r[4])  # total p50 빠른 순

lines = ["| 순위 | 백엔드 | 측정 횟수 | inference p50 | postprocess p50 | total p50 | total p95 | total max | FPS(p50) | FPS(p95) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
lines += [f"| {i} | {b} | {n} | {a:.2f} | {pp:.2f} | {p50:.2f} | {p95:.2f} | {mx:.2f} | {f50:.1f} | {f95:.1f} |"
          for i, (b, n, a, pp, p50, p95, mx, f50, f95) in enumerate(rows, 1)]
table = "\n".join(lines)
print(table)
(folder / "summary.md").write_text(f"# yolo26n 백엔드 비교 ({folder.name})\n\n단위: ms\n\n{table}\n", encoding="utf-8")
print(f"\n저장: {folder / 'summary.md'}")
