"""Experiment 5: real text embeddings (AG News, all-MiniLM-L6-v2, 384-d).

Part 1: same tree as scipy on the first 20k embeddings.
Part 2: timing on all 120k (scipy would need a 58 GB matrix).

usage: python experiments/make_embeddings.py && python experiments/exp5_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastlinkage.core import linkage  # noqa: E402

SCIPY = {"average_cosine": ("average", "cosine"), "ward": ("ward", "euclidean"),
         "average_sqeuclidean": ("average", "sqeuclidean")}


def main():
    E = np.load(ROOT / "data" / "agnews_minilm.npy").astype(np.float64)
    linkage(E[:3000], "average_cosine")  # warm up
    out = {"data": "AG News 120k x 384, all-MiniLM-L6-v2", "same_tree": [], "timing": []}
    X = E[:20_000]
    for method, (sm, metric) in SCIPY.items():
        t0 = time.perf_counter()
        Zs = scipy_linkage(X, sm, metric)
        ts = time.perf_counter() - t0
        Z, st = linkage(X, method, return_stats=True)
        diff = float(np.max(np.abs(Z[:, 2] - Zs[:, 2])))
        ari = min(adjusted_rand_score(fcluster(Zs, k, "maxclust"), fcluster(Z, k, "maxclust"))
                  for k in (2, 5, 20, 100, 1000))
        row = {"n": 20_000, "method": method, "gpu_s": round(st["t_total"], 2), "scipy_s": round(ts, 1),
               "max_abs_height_diff": diff, "min_ari": ari}
        out["same_tree"].append(row)
        print(row, flush=True)
    for method in SCIPY:
        Z, st = linkage(E, method, return_stats=True)
        row = {"n": len(E), "method": method, "gpu_s": round(st["t_total"], 1), "rounds": st["n_rounds"],
               "pair_work_vs_n2": round(st["pair_work_vs_n2"], 2), "t_search": round(st["t_gpu"], 1),
               "t_verify": round(st["t_verify"], 1)}
        if method == "average_cosine":
            row["same_code_cpu_s"] = round(linkage(E, method, device="cpu", return_stats=True)[1]["t_total"], 1)
        out["timing"].append(row)
        print(row, flush=True)
    (ROOT / "results" / "exp5_embeddings.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
