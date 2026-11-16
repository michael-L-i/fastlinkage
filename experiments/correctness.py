"""Does the matrix-free tree match scipy's?

Compares merge heights and the flat clusterings cut from the tree, for each
method, on clustered data and on unstructured data.

usage: python experiments/correctness.py [device]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from fastlinkage import linkage

ROOT = Path(__file__).resolve().parents[1]

SCIPY = {"ward": ("ward", "euclidean"), "average_cosine": ("average", "cosine"),
         "average_sqeuclidean": ("average", "sqeuclidean")}


def datasets():
    rng = np.random.default_rng(0)
    yield "blobs 5k x 16", make_blobs(5_000, 16, centers=30, random_state=0)[0]
    yield "normal 5k x 32", rng.standard_normal((5_000, 32))
    yield "blobs 20k x 32", make_blobs(20_000, 32, centers=100, cluster_std=2.0, random_state=1)[0]


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for name, X in datasets():
        for method, (sm, metric) in SCIPY.items():
            t0 = time.perf_counter()
            Zs = scipy_linkage(X, method=sm, metric=metric)
            ts = time.perf_counter() - t0
            Z, st = linkage(X, method=method, device=device, return_stats=True)
            rel = float(np.max(np.abs(Z[:, 2] - Zs[:, 2]) / np.maximum(Zs[:, 2], 1e-300)))
            ari = min(adjusted_rand_score(fcluster(Zs, kc, "maxclust"), fcluster(Z, kc, "maxclust"))
                      for kc in (2, 5, 20, 100, 1000))
            rows.append({"data": name, "method": method, "max_rel_height_diff": rel, "min_ari": ari,
                         "rounds": st["n_rounds"], "time_s": round(st["t_total"], 3),
                         "scipy_s": round(ts, 2), "device": st["device"]})
            print(f"{name:16s} {method:20s} height diff={rel:.1e} min ARI={ari:.6f} "
                  f"rounds={st['n_rounds']:3d} ours={st['t_total']:.2f}s scipy={ts:.1f}s", flush=True)
    (ROOT / "results" / "correctness.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
