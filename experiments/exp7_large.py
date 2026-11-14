"""Experiment 7: larger runs of the cached version.

Part 1: 100k points, Ward, against fastcluster's exact memory-saving Ward (an
independent implementation; scipy cannot run this size).
Part 2: timings at 200k and 500k points.

usage: python experiments/exp7_large.py
"""
import json
import sys
import time
from pathlib import Path

import fastcluster
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_cache import blobs  # noqa: E402
from fastlinkage.core import METHODS, linkage  # noqa: E402


def main():
    out = {"ward_vs_fastcluster": None, "timing": []}
    linkage(blobs(3000), "ward")
    X = blobs(100_000)
    Z, st = linkage(X, "ward", return_stats=True)
    t0 = time.perf_counter()
    Zf = fastcluster.linkage_vector(X, method="ward")
    tf = time.perf_counter() - t0
    out["ward_vs_fastcluster"] = {"n": 100_000, "ours_s": round(st["t_total"], 2), "fastcluster_s": round(tf, 1),
                                  "max_height_diff": float(np.max(np.abs(Z[:, 2] - Zf[:, 2]))),
                                  "max_height": float(Zf[:, 2].max())}
    print(out["ward_vs_fastcluster"], flush=True)
    for n in (200_000, 500_000):
        X = blobs(n)
        for method in METHODS:
            Z, st = linkage(X, method, return_stats=True)
            row = {"n": n, "method": method, "s": round(st["t_total"], 1), "rounds": st["n_rounds"],
                   "pair_work_vs_n2": round(st["pair_work_vs_n2"], 2), "t_gpu": round(st["t_gpu"], 1),
                   "t_verify": round(st["t_verify"], 1), "t_cache": round(st["t_cache"], 1),
                   "rechecked": st["rechecked"]}
            out["timing"].append(row)
            print(row, flush=True)
    (ROOT / "results" / "exp7_large.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
