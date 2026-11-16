"""Could a low-dimensional bound prune the search in high dimensions?

Rotate the data onto its principal axes and keep the first p coordinates plus
the norm of the rest. For unit-free vectors a and b,

    a . b  <=  head(a) . head(b) + |tail(a)| |tail(b)|

so a (p+1)-dimensional matrix product gives an upper bound on every dot
product (a lower bound on the cosine key). A column can be skipped for a query
when its bound is worse than the query's true nearest neighbour.

This measures, for sample queries on real embeddings, the fraction of columns
that the bound fails to rule out. The exact search would cost about
(p+1)/d + that fraction, relative to brute force.

usage: python experiments/pruning_bound.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    E = np.load(ROOT / "data" / "agnews_minilm.npy").astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    n, d = E.shape
    rng = np.random.default_rng(0)
    q = rng.choice(n, 2000, replace=False)
    # principal axes of the (uncentred) unit vectors
    _, _, Vt = np.linalg.svd(E[rng.choice(n, 20_000, replace=False)], full_matrices=False)
    R = E @ Vt.T                                   # rotated coordinates, same dot products
    dev = "mps"
    Rg = torch.from_numpy(R.astype(np.float32)).to(dev)
    true = Rg[q] @ Rg.T
    true[torch.arange(len(q)), torch.from_numpy(q).to(dev)] = -2.0
    best = true.max(1).values                      # dot with the true nearest neighbour
    rows = []
    for p in (8, 16, 32, 64, 128):
        head = Rg[:, :p]
        tail = (Rg[:, p:] ** 2).sum(1).sqrt()
        ub = head[q] @ head.T + tail[q][:, None] * tail[None, :]
        ub[torch.arange(len(q)), torch.from_numpy(q).to(dev)] = -2.0
        frac = (ub >= best[:, None]).float().mean(1).cpu().numpy()
        rows.append({"p": p, "bound_cost_vs_brute": round((p + 1) / d, 3),
                     "unpruned_mean": float(frac.mean()), "unpruned_median": float(np.median(frac)),
                     "unpruned_p90": float(np.quantile(frac, 0.9))})
        print(rows[-1], flush=True)
    (ROOT / "results" / "pruning_bound.json").write_text(json.dumps(
        {"data": "AG News 120k x 384 (all-MiniLM-L6-v2), cosine, single points", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
