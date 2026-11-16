"""Ward on all 120k embeddings against fastcluster (independent, exact, slow).

The package and the brute-force baseline disagree on this input; this says which is right.

usage: python experiments/ward_embeddings_check.py
"""
import json
import time
from pathlib import Path

import fastcluster
import numpy as np
from brute_force_baseline import linkage as linkage_brute

from fastlinkage import linkage

ROOT = Path(__file__).resolve().parents[1]

E = np.load(ROOT / "data" / "agnews_minilm.npy").astype(np.float64)
Z = linkage(E, "ward")
Z1 = linkage_brute(E, "ward")
t0 = time.perf_counter()
Zf = fastcluster.linkage_vector(E, method="ward")
out = {"n": len(E), "fastcluster_s": round(time.perf_counter() - t0, 1),
       "max_height_diff": float(np.max(np.abs(Z[:, 2] - Zf[:, 2]))),
       "brute_force_max_height_diff": float(np.max(np.abs(Z1[:, 2] - Zf[:, 2]))),
       "brute_force_merges_differing": int(np.sum(np.abs(Z1[:, 2] - Zf[:, 2]) > 1e-9)),
       "max_height": float(Zf[:, 2].max())}
print(out)
(ROOT / "results" / "ward_embeddings_check.json").write_text(json.dumps(out, indent=1))
