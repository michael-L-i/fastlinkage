"""Experiment 8: Ward on all 120k embeddings against fastcluster (independent, exact, slow).

Version 1 and the cached version disagree on this input; this says which is right.

usage: python experiments/exp8_ward_check_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

import fastcluster
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastlinkage.core import linkage  # noqa: E402
from baseline_v1 import linkage as linkage_v1  # noqa: E402

E = np.load(ROOT / "data" / "agnews_minilm.npy").astype(np.float64)
Z = linkage(E, "ward")
Z1 = linkage_v1(E, "ward")
t0 = time.perf_counter()
Zf = fastcluster.linkage_vector(E, method="ward")
out = {"n": len(E), "fastcluster_s": round(time.perf_counter() - t0, 1),
       "cached_max_height_diff": float(np.max(np.abs(Z[:, 2] - Zf[:, 2]))),
       "v1_max_height_diff": float(np.max(np.abs(Z1[:, 2] - Zf[:, 2]))),
       "v1_merges_differing": int(np.sum(np.abs(Z1[:, 2] - Zf[:, 2]) > 1e-9)),
       "max_height": float(Zf[:, 2].max())}
print(out)
(ROOT / "results" / "exp8_ward_check_embeddings.json").write_text(json.dumps(out, indent=1))
