"""Exact hierarchical clustering without the n x n distance matrix.

For some linkages the distance between two clusters depends only on a small
summary of each cluster (size, mean vector, sometimes one scalar). Those
linkages can be run from the summaries alone:

  ward                  sqrt(2 |A||B| / (|A|+|B|)) * ||mean(A) - mean(B)||
  average, cosine       1 - mean(A) . mean(B)          (rows scaled to unit length)
  average, sqeuclidean  ||mean(A) - mean(B)||^2 + var(A) + var(B)
                        where var is the mean squared distance to the cluster mean

All three are "reducible": merging two clusters never brings the result closer
to a third cluster than both parts were. Three consequences:

  1. Every pair of clusters that are each other's nearest neighbour can be
     merged in the same round, and the tree is the same one the usual
     one-merge-at-a-time algorithm builds.
  2. A cluster's nearest neighbour stays valid until that neighbour itself is
     merged.
  3. Neighbour lists survive merges (`cache=True`). Say cluster c knows a list
     of clusters with their exact distances, and a radius r such that every
     cluster outside the list is at least r away. After any merges, a cluster
     that contains no list member is still at least r away. So c's new nearest
     neighbour is the nearest of the list members' current clusters, as long
     as that one is within r. A new cluster A+B inherits the lists of A and B
     with a radius computed from r_A and r_B. Only when a list runs dry does
     the cluster go back to the full search.

The full search is one big matrix product, which runs on the GPU in float32.
It returns a few candidates per cluster and the smallest float32 distance
among all the other columns. The CPU computes the candidates' distances in
float64. A candidate is accepted only if it beats the other columns by more
than the float32 rounding bound; otherwise every column inside the bound is
fetched and checked in float64. Merge decisions and heights are float64.
"""
import time

import numpy as np
import torch

from ._clusters import Clusters
from ._distance import METHODS, height
from ._search import Searcher
from ._tree import to_scipy

__all__ = ["METHODS", "linkage"]


def _default_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _identical_rows(X):
    """For each row, the index of the first row equal to it."""
    n, d = X.shape
    probe = X @ np.random.default_rng(0).standard_normal(d)
    if len(np.unique(probe)) == n:               # cheap test: no two rows can be equal
        return np.arange(n)
    _, first, inverse = np.unique(X, axis=0, return_index=True, return_inverse=True)
    return first[inverse.ravel()]


def linkage(X, method="ward", device=None, k=16, segs=256, budget=1 << 25, cache=True,
            width=None, return_stats=False):
    """Exact linkage matrix in scipy's format, without the n x n distance matrix.

    X: (n, d) array of points.
    method: "ward", "average_cosine" or "average_sqeuclidean".
    device: "mps", "cuda" or "cpu" (default: the GPU if there is one).
    k: candidates per cluster that come back from a full search.
    segs: about how many segments the columns are cut into for the search.
    budget: number of float32 elements per distance block on the device.
    cache: keep each cluster's neighbour list across merges and use it before
       falling back to a full search.
    width: entries kept per neighbour list (default k).
    return_stats: also return a dict of timings and work counts.

    Returns Z, an (n - 1, 4) array as scipy.cluster.hierarchy.linkage returns.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if device is None:
        device = _default_device()
    t_start = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    if method == "average_cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError("cosine distance is undefined for zero rows")
        X = X / norms
    else:
        X = X - X.mean(axis=0)  # distances do not change; float32 keeps more digits

    cl = Clusters(X, method, device, width or k)
    stats = cl.stats
    active = np.arange(n, dtype=np.int64)
    m_a, m_b, m_h = [], [], []      # the merges: one member of each side, and the height

    # Identical rows merge first, at height zero, in any order. Do that up
    # front: they would otherwise be exact ties all the way through.
    first = _identical_rows(X)
    extra = np.nonzero(first != np.arange(n))[0]
    if len(extra):
        m_a.append(first[extra])
        m_b.append(extra)
        m_h.append(np.zeros(len(extra)))
        active = np.unique(first)
        cl.size[active] = np.bincount(first, minlength=n)[active]

    fresh = pa = pb = None          # clusters made in the last round, and their parents
    while len(active) > 1:
        m = len(active)
        if cache and fresh is not None:
            cl.refresh_from_lists(active, fresh, pa, pb)
        todo = active[~cl.valid[active]]
        if len(todo):
            t0 = time.perf_counter()
            searcher = Searcher(method, cl.G, cl.size, cl.var, cl.norm, active, k, segs, budget, device)
            stats["t_gpu"] += time.perf_counter() - t0
            cl.full_search(searcher, todo)
            del searcher
        a = cl.reciprocal_pairs(active)
        if len(a) == 0:
            a = cl.tied_pairs(active)
        b = cl.nn_id[a]
        m_a.append(cl.rep[a])
        m_b.append(cl.rep[b])
        m_h.append(height(method, cl.nn_key[a]))
        new = cl.merge(a, b)
        keep = active[~cl.gone[active]]
        cl.valid[keep[cl.gone[cl.nn_id[keep]]]] = False   # their nearest neighbour was merged away
        stats["rounds"].append((m, len(todo), len(a)))
        active = np.concatenate((keep, new))
        fresh, pa, pb = new, a, b

    Z = to_scipy(np.concatenate(m_a), np.concatenate(m_b), np.concatenate(m_h), n)
    stats["t_total"] = time.perf_counter() - t_start
    stats["n_rounds"] = len(stats["rounds"])
    stats["pair_work_vs_n2"] = stats["pair_evals"] / (n * n)
    stats["device"] = device
    return (Z, stats) if return_stats else Z
