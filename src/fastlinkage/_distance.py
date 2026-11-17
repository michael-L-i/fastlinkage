"""Cluster-to-cluster distances from cluster summaries, in float64.

Each supported linkage has a "key": a number that orders pairs of clusters the
same way the linkage distance does, and that follows from each cluster's size,
mean vector and (for one linkage) a variance.

  ward                  |A||B| / (|A|+|B|) * ||mean(A) - mean(B)||^2
                        (the merge height is sqrt(2 * key))
  average_cosine        1 - mean(A) . mean(B)          (rows scaled to unit length)
  average_sqeuclidean   ||mean(A) - mean(B)||^2 + var(A) + var(B)
                        where var is the mean squared distance to the cluster mean
"""
import numpy as np
import torch

METHODS = ("ward", "average_cosine", "average_sqeuclidean")


def exact_key(method, Ct, size, var, q, c):
    """Key between each cluster id in q (n,) and its candidates c (n, k).

    Ct holds the cluster means as a float64 tensor (one row per cluster id),
    on the CPU or on a device that supports float64. size and var are the
    cluster sizes and variances (numpy)."""
    cq = Ct[torch.from_numpy(q).to(Ct.device)]
    cc = Ct[torch.from_numpy(c).to(Ct.device)]
    if method == "average_cosine":
        return 1.0 - torch.einsum("nd,nkd->nk", cq, cc).cpu().numpy()
    diff = cc - cq[:, None, :]
    d2 = torch.einsum("nkd,nkd->nk", diff, diff).cpu().numpy()
    sq = size[q][:, None]
    if method == "ward":
        return d2 * (sq * size[c] / (sq + size[c]))
    return d2 + var[q][:, None] + var[c]


def height(method, key):
    """Merge height, as scipy reports it, from a key."""
    key = np.maximum(key, 0.0)  # duplicate rows can give -1e-16 from rounding
    return np.sqrt(2.0 * key) if method == "ward" else key
