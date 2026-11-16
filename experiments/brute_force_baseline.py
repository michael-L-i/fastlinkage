"""Exact hierarchical clustering without the n x n distance matrix.

For some linkages the distance between two clusters depends only on a small
summary of each cluster (size, mean vector, sometimes one scalar). Those
linkages can be run from the summaries alone:

  ward                  sqrt(2 |A||B| / (|A|+|B|)) * ||mean(A) - mean(B)||
  average, cosine       1 - mean(A) . mean(B)          (rows scaled to unit length)
  average, sqeuclidean  ||mean(A) - mean(B)||^2 + var(A) + var(B)
                        where var is the mean squared distance to the cluster mean

All three are "reducible": merging two clusters never brings the result closer
to a third cluster than both parts were. Two consequences:

  1. Every pair of clusters that are each other's nearest neighbour can be
     merged in the same round, and the tree is the same one the usual
     one-merge-at-a-time algorithm builds.
  2. A cluster's nearest neighbour stays valid until that neighbour itself is
     merged. A cluster that lost its neighbour ("orphan") only has to search
     again when some other cluster names it as nearest neighbour, because it
     cannot be in a reciprocal pair otherwise (`lazy=True`).

The nearest-neighbour search is one big matrix product, which runs on the GPU
in float32. It returns a few candidates per cluster; the CPU then picks the
true nearest among them in float64, so merge decisions and heights are float64.
(If two neighbours tie to within float32 rounding and fall in the same column
block, the float64 check sees only one of them.)
"""
import time

import numpy as np
import torch

METHODS = ("ward", "average_cosine", "average_sqeuclidean")
MAX_SIZE_GROUPS = 24


def _exact_key(method, C, size, var, q, c):
    """Float64 ordering key between each cluster id in q (n,) and its candidates c (n, k)."""
    Ct = torch.from_numpy(C)                      # shares memory; threaded on the CPU
    cq = Ct[torch.from_numpy(q)]
    cc = Ct[torch.from_numpy(c)]
    if method == "average_cosine":
        return 1.0 - torch.einsum("nd,nkd->nk", cq, cc).numpy()
    diff = cc - cq[:, None, :]
    d2 = torch.einsum("nkd,nkd->nk", diff, diff).numpy()
    sq = size[q][:, None]
    if method == "ward":
        return d2 * (sq * size[c] / (sq + size[c]))
    return d2 + var[q][:, None] + var[c]


def _height(method, key):
    key = np.maximum(key, 0.0)  # duplicate rows can give -1e-16 from rounding
    return np.sqrt(2.0 * key) if method == "ward" else key


class _Searcher:
    """Nearest-neighbour candidates among the active clusters, on the device.

    The float32 key comes out of a single matrix product: the squared distance
    (plus the variance term, for average linkage) is folded into two extra
    columns. The columns are cut into about k segments and the best column of
    each segment is a candidate, which costs one argmin pass per segment.

    Ward's key also has the factor |A||B| / (|A|+|B|). While there are few
    distinct cluster sizes, the columns are sorted by size so every segment has
    one size; inside a segment the factor is the same for all columns, so the
    squared distance alone picks the segment's best. Otherwise the factor is
    applied to the whole block.
    """

    def __init__(self, method, C, size, var, active, k, budget, device):
        self.method = method
        self.budget = budget
        self.device = device
        m = len(active)
        self.m = m
        s_act = size[active]
        self.ward_div = False
        order = np.arange(m)
        if method == "ward":
            uniq, counts = np.unique(s_act, return_counts=True)
            if len(uniq) <= MAX_SIZE_GROUPS:
                order = np.argsort(s_act, kind="stable")
                bounds = [0]
                for c in counts:  # split each size group so there are about k segments in all
                    parts = max(1, int(round(k * c / m)))
                    step = -(-c // parts)
                    start = bounds[-1]
                    bounds.extend(min(start + c, start + step * (i + 1)) for i in range(parts))
                self.bounds = sorted(set(bounds))
            else:
                self.ward_div = True
        if method != "ward" or self.ward_div:
            kk = max(1, min(k, m))
            self.bounds = sorted(set([m * i // kk for i in range(kk)] + [m]))
        self.cols = active[order]                 # cluster id of each device column
        self.pos = {}
        Cg = torch.from_numpy(C[self.cols].astype(np.float32)).to(device)
        if method == "average_cosine":
            self.A_t = (-Cg).T.contiguous()       # key = -dot
            self.Q = Cg
        else:
            n2 = (Cg * Cg).sum(1, keepdim=True)
            one = torch.ones_like(n2)
            extra = n2
            if method == "average_sqeuclidean":
                extra = n2 + torch.from_numpy(var[self.cols].astype(np.float32)).to(device)[:, None]
            self.A_t = torch.cat((-2.0 * Cg, one, extra), 1).T.contiguous()
            self.Q = torch.cat((Cg, n2, one), 1)  # Q @ A_t = squared distance (+ variance)
        if self.ward_div:
            self.inv = 1.0 / torch.from_numpy(s_act[order].astype(np.float32)).to(device)
        self.col_of = np.full(len(size), -1, np.int64)
        self.col_of[self.cols] = np.arange(m)

    def _segment_argmin(self, key):
        b = self.bounds
        return torch.stack([key[:, b[i]:b[i + 1]].argmin(1) + b[i] for i in range(len(b) - 1)], 1)

    def _brute_keys(self, qc):
        key = self.Q[qc] @ self.A_t
        if self.ward_div:
            key.div_(self.inv[qc][:, None] + self.inv[None, :])
        key[torch.arange(len(qc), device=self.device), qc] = float("inf")
        return key

    def candidates(self, query):
        """Cluster ids of the candidate neighbours of each cluster id in `query`."""
        qcol = torch.from_numpy(self.col_of[query]).to(self.device)
        rows = max(1, self.budget // self.m)
        out = []
        for s in range(0, len(query), rows):
            qc = qcol[s:s + rows]
            out.append(self._segment_argmin(self._brute_keys(qc)))
        return self.cols[torch.cat(out).cpu().numpy()]


def linkage(X, method="ward", device=None, k=8, budget=1 << 25, lazy=False, return_stats=False):
    """Exact linkage matrix in scipy's format, for the methods in METHODS.

    device: "mps", "cuda" or "cpu" (default: the GPU if there is one).
    k: about how many candidates per cluster go from the float32 search to the
       float64 check.
    budget: number of float32 elements per distance block on the device.
    lazy: orphans search again only when another cluster points at them. Fewer
       pair evaluations (about 25%) but many small searches, which is slower on a GPU.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape
    if method == "average_cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError("cosine distance is undefined for zero rows")
        X = X / norms
    else:
        X = X - X.mean(axis=0)  # distances do not change; float32 keeps more digits

    total = 2 * n - 1
    C = np.empty((total, d))
    C[:n] = X
    size = np.ones(total)
    var = np.zeros(total)
    rep = np.arange(total)          # one member point of each cluster
    nn_id = np.full(total, -1, np.int64)
    nn_key = np.full(total, np.inf)
    valid = np.zeros(total, bool)   # nn_id is the true nearest neighbour right now
    gone = np.zeros(total, bool)

    active = np.arange(n, dtype=np.int64)
    fresh = active.copy()           # clusters that have never searched
    next_id = n
    m_a, m_b, m_h = [], [], []
    stats = {"rounds": [], "t_gpu": 0.0, "t_verify": 0.0, "pair_evals": 0}

    def search(searcher, query):
        if len(query) == 0:
            return
        t0 = time.perf_counter()
        cand = searcher.candidates(query)
        stats["t_gpu"] += time.perf_counter() - t0
        stats["pair_evals"] += len(query) * searcher.m
        t0 = time.perf_counter()
        step = max(1, (1 << 25) // (cand.shape[1] * d))
        for s in range(0, len(query), step):
            q = query[s:s + step]
            c = cand[s:s + step]
            keys = _exact_key(method, C, size, var, q, c)
            keys[c == q[:, None]] = np.inf           # a segment can return the cluster itself
            best = np.lexsort((c, keys), axis=1)[:, 0]   # smallest key, then smallest id
            r = np.arange(len(q))
            nn_id[q] = c[r, best]
            nn_key[q] = keys[r, best]
        valid[query] = True
        stats["t_verify"] += time.perf_counter() - t0

    def reciprocal():
        ok = valid[active]
        a = active[ok]
        b = nn_id[a]
        keep = valid[b] & (nn_id[b] == a) & (a < b)
        return a[keep]

    while len(active) > 1:
        m = len(active)
        t0 = time.perf_counter()
        searcher = _Searcher(method, C, size, var, active, k, budget, device)
        stats["t_gpu"] += time.perf_counter() - t0
        searched = len(fresh)
        search(searcher, fresh)
        if lazy:
            # Follow the chain: a newly searched cluster may point at another
            # orphan, which then has to search too. Stop when no cluster with a
            # known neighbour points at an orphan.
            while True:
                orphans = active[~valid[active]]
                if len(orphans) == 0:
                    break
                wanted = np.zeros(total, bool)
                wanted[nn_id[active[valid[active]]]] = True
                todo = orphans[wanted[orphans]]
                if len(todo) == 0:
                    break
                searched += len(todo)
                search(searcher, todo)
        else:
            todo = active[~valid[active]]
            searched += len(todo)
            search(searcher, todo)
        a = reciprocal()
        if len(a) == 0 and lazy:                 # nothing reciprocal: search every orphan
            rest = active[~valid[active]]
            searched += len(rest)
            search(searcher, rest)
            a = reciprocal()
        if len(a) == 0:                          # only with exact ties: merge the closest pair
            x = active[int(np.argmin(nn_key[active]))]
            y = nn_id[x]
            nn_id[y] = x
            a = np.array([min(x, y)])
            nn_id[a[0]] = max(x, y)
            nn_key[a[0]] = nn_key[x]
        del searcher
        b = nn_id[a]
        p = len(a)
        new = np.arange(next_id, next_id + p)
        next_id += p
        sa, sb = size[a], size[b]
        sn = sa + sb
        C[new] = (C[a] * sa[:, None] + C[b] * sb[:, None]) / sn[:, None]
        if method == "average_sqeuclidean":
            da = np.einsum("ij,ij->i", C[a] - C[new], C[a] - C[new])
            db = np.einsum("ij,ij->i", C[b] - C[new], C[b] - C[new])
            var[new] = (sa * (var[a] + da) + sb * (var[b] + db)) / sn
        size[new] = sn
        rep[new] = rep[a]
        m_a.append(rep[a])
        m_b.append(rep[b])
        m_h.append(_height(method, nn_key[a]))
        gone[a] = True
        gone[b] = True
        keep = active[~gone[active]]
        valid[keep[gone[nn_id[keep]]]] = False   # their nearest neighbour was merged away
        stats["rounds"].append((m, searched, p))
        active = np.concatenate((keep, new))
        fresh = new

    Z = _label(np.concatenate(m_a), np.concatenate(m_b), np.concatenate(m_h), n)
    stats["t_total"] = time.perf_counter() - t_start
    stats["n_rounds"] = len(stats["rounds"])
    stats["pair_work_vs_n2"] = stats["pair_evals"] / (n * n)
    stats["device"] = device
    return (Z, stats) if return_stats else Z


def _label(ra, rb, h, n):
    """Sort merges by height and number the clusters the way scipy does."""
    order = np.argsort(h, kind="stable")
    parent = np.arange(n)
    label = np.arange(n)
    csize = np.ones(n, np.int64)
    Z = np.empty((n - 1, 4))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    ra = ra.tolist()
    rb = rb.tolist()
    for i, j in enumerate(order.tolist()):
        x = find(ra[j])
        y = find(rb[j])
        lx, ly = int(label[x]), int(label[y])
        Z[i, 0] = min(lx, ly)
        Z[i, 1] = max(lx, ly)
        Z[i, 2] = h[j]
        s = csize[x] + csize[y]
        Z[i, 3] = s
        parent[y] = x
        label[x] = n + i
        csize[x] = s
    return Z
