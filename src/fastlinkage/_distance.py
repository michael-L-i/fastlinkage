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

# scipy / scikit-learn spelling (linkage, metric) -> method
_PAIRS = {("ward", "euclidean"): "ward", ("average", "cosine"): "average_cosine",
          ("average", "sqeuclidean"): "average_sqeuclidean"}
_SUPPORTED = ("supported: 'ward' (Euclidean), and 'average' with metric='cosine' or "
              "metric='sqeuclidean'. Other linkages need the full distance matrix; "
              "use scipy.cluster.hierarchy.linkage for those")


def resolve_method(method, metric=None):
    """One of METHODS from either spelling: a name in METHODS, or scipy's
    (method, metric) pair such as ("average", "cosine")."""
    if metric is None:
        if method in METHODS:
            return method
        if method == "average":
            raise ValueError("average linkage needs metric='cosine' or metric='sqeuclidean' "
                             "(plain Euclidean average linkage needs the full distance matrix)")
    elif (method, metric) in _PAIRS:
        return _PAIRS[(method, metric)]
    elif method in METHODS and _PAIRS.get((method.split("_")[0], metric)) == method:
        return method
    what = f"method={method!r}" + ("" if metric is None else f" with metric={metric!r}")
    raise ValueError(f"{what} is not supported; {_SUPPORTED}")


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
