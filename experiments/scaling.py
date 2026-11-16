"""Experiment 2: timing and round counts as n grows (modest sizes).

Baselines: scipy (needs the n x n matrix, so 20k points only) and
fastcluster.linkage_vector, the standard memory-saving exact Ward on the CPU
(run up to 50k; the first version of this script measured 129 s at 100k).

usage: python experiments/exp2_scaling.py
"""
import json
import sys
import time
from pathlib import Path

import fastcluster
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastlinkage.core import linkage  # noqa: E402

SCIPY = {"ward": ("ward", "euclidean"), "average_cosine": ("average", "cosine"),
         "average_sqeuclidean": ("average", "sqeuclidean")}


def main():
    rows = []
    linkage(np.random.default_rng(0).standard_normal((2000, 32)))  # warm up the GPU
    for n in (20_000, 50_000, 100_000, 200_000, 500_000):
        X = make_blobs(n, 32, centers=n // 200, cluster_std=2.0, random_state=0)[0]
        for method in SCIPY:
            if n == 500_000 and method == "average_sqeuclidean":
                continue  # keep the largest run short
            Z, st = linkage(X, method=method, return_stats=True)
            row = {"n": n, "method": method, "gpu_s": round(st["t_total"], 2),
                   "rounds": st["n_rounds"], "pair_work_vs_n2": round(st["pair_work_vs_n2"], 2),
                   "t_search": round(st["t_gpu"], 2), "t_verify": round(st["t_verify"], 2)}
            if n == 20_000:
                t0 = time.perf_counter()
                scipy_linkage(X, method=SCIPY[method][0], metric=SCIPY[method][1])
                row["scipy_s"] = round(time.perf_counter() - t0, 2)
            if n == 50_000:
                row["same_code_cpu_s"] = round(linkage(X, method=method, device="cpu",
                                                       return_stats=True)[1]["t_total"], 2)
            if method == "ward" and n <= 50_000:
                t0 = time.perf_counter()
                fastcluster.linkage_vector(X, method="ward")
                row["fastcluster_s"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            print(row, flush=True)
    (ROOT / "results" / "exp2_scaling.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
