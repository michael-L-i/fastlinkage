"""Where the time goes in one run: the package's own stage timers, then the
functions that took longest.

usage: python experiments/profile_run.py [n] [d] [method] [device]
"""
import cProfile
import pstats
import sys

import numpy as np
import torch
from sklearn.datasets import make_blobs

from fastlinkage import linkage


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    method = sys.argv[3] if len(sys.argv) > 3 else "ward"
    device = sys.argv[4] if len(sys.argv) > 4 else None
    X = make_blobs(n, d, centers=n // 200, cluster_std=2.0, random_state=0)[0]
    linkage(np.random.default_rng(0).standard_normal((3000, d)), method, device=device)   # warm up

    best = None
    for _ in range(3):
        st = linkage(X, method, device=device, return_stats=True)[1]
        if best is None or st["t_total"] < best["t_total"]:
            best = st
    other = best["t_total"] - best["t_gpu"] - best["t_verify"] - best["t_cache"]
    print(f"{n} x {d} {method} on {best['device']}: total {best['t_total']:.2f} s = search {best['t_gpu']:.2f}"
          f" + float64 check {best['t_verify']:.2f} + lists {best['t_cache']:.2f} + other {other:.2f}"
          f" | rounds {best['n_rounds']}, pair work {best['pair_work_vs_n2']:.2f} n^2")

    if best["device"].startswith("cuda"):
        torch.cuda.synchronize()
    profile = cProfile.Profile()
    profile.enable()
    linkage(X, method, device=device)
    profile.disable()
    pstats.Stats(profile).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()
