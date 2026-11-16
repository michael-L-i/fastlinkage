"""Neighbour lists against the brute-force baseline (a full search every round).

Same tree as scipy, and how much full-search work the neighbour lists save.

usage: python experiments/neighbour_lists.py [quick]
"""
import json
import sys
from pathlib import Path

import numpy as np
from brute_force_baseline import linkage as linkage_brute
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.metrics import adjusted_rand_score

from fastlinkage import METHODS, linkage

ROOT = Path(__file__).resolve().parents[1]

SCIPY = {"average_cosine": ("average", "cosine"), "ward": ("ward", "euclidean"),
         "average_sqeuclidean": ("average", "sqeuclidean")}


def blobs(n, d=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(200, d)) * 4
    return centres[rng.integers(0, 200, n)] + rng.normal(size=(n, d))


def row_for(name, X, method, check_scipy, **kw):
    Z, st = linkage(X, method, return_stats=True, **kw)
    Z1, s1 = linkage_brute(X, method, return_stats=True)
    row = {"data": name, "n": len(X), "method": method, **kw,
           "fastlinkage_s": round(st["t_total"], 2), "brute_force_s": round(s1["t_total"], 2),
           "work": round(st["pair_work_vs_n2"], 2), "work_brute_force": round(s1["pair_work_vs_n2"], 2),
           "fresh_hit": round(st["fresh_hits"] / max(1, st["fresh_total"]), 3),
           "orphan_hit": round(st["orphan_hits"] / max(1, st["orphan_total"]), 3),
           "rechecked": st["rechecked"], "rounds": st["n_rounds"],
           "t_gpu": round(st["t_gpu"], 2), "t_verify": round(st["t_verify"], 2),
           "t_cache": round(st["t_cache"], 2),
           "max_height_diff_brute_force": float(np.max(np.abs(Z[:, 2] - Z1[:, 2])))}
    if check_scipy:
        Zs = scipy_linkage(X, *SCIPY[method])
        row["max_height_diff_scipy"] = float(np.max(np.abs(Z[:, 2] - Zs[:, 2])))
        row["min_ari_scipy"] = min(adjusted_rand_score(fcluster(Zs, c, "maxclust"), fcluster(Z, c, "maxclust"))
                                   for c in (2, 5, 20, 100, 1000))
    print(row, flush=True)
    return row


def main():
    quick = len(sys.argv) > 1
    E = np.load(ROOT / "data" / "agnews_minilm.npy").astype(np.float64)
    linkage(E[:3000], "average_cosine")
    linkage_brute(E[:3000], "average_cosine")
    rows = []
    for method in METHODS:
        rows.append(row_for("blobs32", blobs(20_000), method, True))
        rows.append(row_for("agnews384", E[:20_000], method, True))
    if not quick:
        for method in METHODS:
            rows.append(row_for("blobs32", blobs(100_000), method, False))
            rows.append(row_for("agnews384", E, method, False))
    (ROOT / "results" / ("neighbour_lists_quick.json" if quick else "neighbour_lists.json")).write_text(
        json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
