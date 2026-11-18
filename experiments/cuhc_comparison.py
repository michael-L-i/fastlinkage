"""fastlinkage against cuhc on the same NVIDIA GPU.

cuhc (https://github.com/shangar21/cuhc) is the closest existing package: exact
Ward on the GPU by the same reciprocal-nearest-neighbour rounds, in float32,
with a hand-written CUDA kernel. For average linkage it builds the n x n
matrix on the GPU.

Measured here:
  ward             time for both; largest merge-height difference from a
                   float64 reference (scipy at 20,000 points, fastcluster at
                   100,000 with --reference)
  average, cosine  time for both until cuhc's matrix no longer fits
  budget           fastlinkage with larger distance blocks

usage: python experiments/cuhc_comparison.py [--reference]
Needs CUDA, `pip install cuhc` (builds from source, needs nvcc and CMake).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.datasets import make_blobs

from fastlinkage import linkage

ROOT = Path(__file__).resolve().parents[1]


def best_of(f, repeats=3):
    """Best wall time of a few runs, and the last result."""
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = f()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return round(min(times), 3), out


def rel_diff(Z, ref):
    return float(np.max(np.abs(Z[:, 2] - ref[:, 2]) / np.maximum(ref[:, 2], 1e-300)))


def datasets():
    rng = np.random.default_rng(0)
    yield "normal", 20_000, 32, rng.standard_normal((20_000, 32))          # cuhc's own benchmark data
    yield "normal", 100_000, 32, rng.standard_normal((100_000, 32))
    for n, d in ((20_000, 32), (100_000, 32), (200_000, 32), (100_000, 384)):
        yield "blobs", n, d, make_blobs(n, d, centers=n // 200, cluster_std=2.0, random_state=0)[0]


def main():
    import cuhc

    reference = "--reference" in sys.argv
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuhc": getattr(cuhc, "__version__", "?"), "ward": [], "average_cosine": [], "budget": []}
    warm = np.random.default_rng(1).standard_normal((3000, 32))
    linkage(warm)
    cuhc.linkage(warm.astype(np.float32), method="ward")

    for name, n, d, X in datasets():
        X32 = X.astype(np.float32)
        row = {"data": name, "n": n, "d": d}
        row["fastlinkage_s"], Zf = best_of(lambda X=X: linkage(X, "ward"))
        row["cuhc_s"], Zc = best_of(lambda X32=X32: cuhc.linkage(X32, method="ward"))
        row["height_rel_diff_between"] = rel_diff(Zc, Zf)
        ref = None
        if n == 20_000:
            ref, row["reference"] = scipy_linkage(X, "ward"), "scipy"
        elif reference and (name, n, d) == ("blobs", 100_000, 32):
            import fastcluster
            ref, row["reference"] = fastcluster.linkage_vector(X, method="ward"), "fastcluster"
        if ref is not None:
            row["fastlinkage_rel_diff"] = rel_diff(Zf, ref)
            row["cuhc_rel_diff"] = rel_diff(Zc, ref)
            # float64 reference from the float32 copy of the data: separates
            # cuhc's input rounding from its arithmetic
            if n == 20_000:
                row["cuhc_rel_diff_float32_input"] = rel_diff(Zc, scipy_linkage(X32.astype(np.float64), "ward"))
        out["ward"].append(row)
        print(row, flush=True)

    for n in (20_000, 40_000, 100_000):
        X = make_blobs(n, 384, centers=n // 200, cluster_std=2.0, random_state=0)[0]
        row = {"n": n, "d": 384}
        row["fastlinkage_s"], Zf = best_of(lambda X=X: linkage(X, "average", metric="cosine"), repeats=2)
        try:
            row["cuhc_s"], Zc = best_of(
                lambda X=X: cuhc.linkage(X.astype(np.float32), method="average", metric="cosine"), repeats=2)
            row["height_rel_diff_between"] = rel_diff(Zc, Zf)
        except Exception as e:                       # the n x n matrix does not fit
            row["cuhc_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        if n == 20_000:
            ref = scipy_linkage(X, "average", metric="cosine")
            row["fastlinkage_rel_diff"] = rel_diff(Zf, ref)
            if "cuhc_s" in row:
                row["cuhc_rel_diff"] = rel_diff(Zc, ref)
        out["average_cosine"].append(row)
        print(row, flush=True)

    for n, d in ((100_000, 32), (100_000, 384)):
        X = make_blobs(n, d, centers=n // 200, cluster_std=2.0, random_state=0)[0]
        for shift in (25, 27, 28, 29):
            t, _ = best_of(lambda X=X, shift=shift: linkage(X, "ward", budget=1 << shift), repeats=2)
            row = {"n": n, "d": d, "budget": f"1<<{shift}", "fastlinkage_s": t}
            out["budget"].append(row)
            print(row, flush=True)

    (ROOT / "results" / "cuhc_comparison.json").write_text(json.dumps(out, indent=1))
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
