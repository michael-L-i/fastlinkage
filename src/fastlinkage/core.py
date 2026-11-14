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
     with radius min(r_A, r_B). Only when a list runs dry does the cluster go
     back to the full search.

The full search is one big matrix product, which runs on the GPU in float32.
It returns a few candidates per cluster and the smallest float32 distance
among all the other columns. The CPU computes the candidates' distances in
float64. A candidate is accepted only if it beats the other columns by more
than the float32 rounding bound; otherwise every column inside the bound is
fetched and checked in float64. Merge decisions and heights are float64.
"""
import time
import warnings

import numpy as np
import torch

METHODS = ("ward", "average_cosine", "average_sqeuclidean")
MAX_SIZE_GROUPS = 24
U32 = 6e-8  # float32 unit roundoff


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
    (plus the variance term, for average linkage) is folded into extra columns.
    The columns are cut into a few hundred equal segments and one pass takes
    the minimum of each. The k segments with the smallest minima supply the
    candidates. A second look at just those k segments finds their runners-up,
    which gives a lower bound on every column that is not a candidate.

    Ward's key also has the factor |A||B| / (|A|+|B|). While there are few
    distinct cluster sizes, the columns are sorted by size and padded so every
    segment holds one size; the factor is then applied to the segment minima.
    Otherwise it is applied to the whole block.
    """

    BIG = 1e30

    def __init__(self, method, G, size, var, norm, active, k, segs, budget, device):
        self.method = method
        self.device = device
        self.size, self.var, self.norm = size, var, norm
        m = len(active)
        self.m = m
        d = G.shape[1]
        s_act = size[active]
        self.ward_div = False
        self.grouped = False
        w = max(1, -(-m // segs))
        self.w = w
        if method == "ward":
            uniq, counts = np.unique(s_act, return_counts=True)
            if len(uniq) <= MAX_SIZE_GROUPS:
                self.grouped = True
            else:
                self.ward_div = True
        if self.grouped:
            order = np.argsort(s_act, kind="stable")
            nseg = -(-counts // w)
            cols = np.full(int(nseg.sum()) * w, -1, np.int64)
            start = np.r_[0, np.cumsum(nseg[:-1])] * w
            src = 0
            for st, c in zip(start, counts):
                cols[st:st + c] = active[order[src:src + c]]
                src += c
            seg_size = np.repeat(uniq, nseg).astype(np.float32)
            self.seg_size64 = np.repeat(uniq, nseg).astype(np.float64)
            self.seg_size = torch.from_numpy(seg_size).to(device)
        else:
            cols = np.full(-(-m // w) * w, -1, np.int64)
            cols[:m] = active
        self.cols = cols                          # cluster id of each device column, -1 = padding
        self.S = len(cols) // w
        self.kk = min(k, self.S)
        self.rows = max(1, budget // len(cols))
        real = cols >= 0
        Cg = G.index_select(0, torch.from_numpy(np.maximum(cols, 0)).to(device))
        if not real.all():
            Cg = Cg * torch.from_numpy(real.astype(np.float32)).to(device)[:, None]
        pad = torch.from_numpy(np.where(real, 0.0, self.BIG).astype(np.float32)).to(device)[:, None]
        one = torch.ones_like(pad)
        if method == "average_cosine":
            self.A_t = torch.cat((-Cg, pad), 1).T.contiguous()     # key = -dot
            self.Q = torch.cat((Cg, one), 1)
        else:
            n2 = (Cg * Cg).sum(1, keepdim=True)
            extra = n2 + pad
            if method == "average_sqeuclidean":
                vp = np.zeros(len(cols), np.float32)
                vp[real] = var[cols[real]]
                extra = extra + torch.from_numpy(vp).to(device)[:, None]
            self.A_t = torch.cat((-2.0 * Cg, one, extra), 1).T.contiguous()
            self.Q = torch.cat((Cg, n2, one), 1)  # Q @ A_t = squared distance (+ column variance)
        if self.ward_div:
            sp = np.ones(len(cols), np.float32)
            sp[real] = s_act
            self.inv = 1.0 / torch.from_numpy(sp).to(device)
        self.col_of = np.full(len(size), -1, np.int64)
        self.col_of[cols[real]] = np.nonzero(real)[0]
        # float32 rounding bound on a key, relative to the sum of the absolute terms
        self.gamma = 4.0 * (d + 4) * U32
        self.amax = float(norm[active].max())
        self.vmax = float(var[active].max())

    def _err(self, query):
        if self.method == "average_cosine":
            return np.full(len(query), self.gamma)
        return self.gamma * ((self.norm[query] + self.amax) ** 2 + self.var[query] + self.vmax)

    def _shift(self, query):
        """exact key = float32 key + shift, for the two average linkages."""
        if self.method == "average_cosine":
            return np.ones(len(query))
        if self.method == "average_sqeuclidean":
            return self.var[query]
        return np.zeros(len(query))

    def _keys(self, qc):
        key = self.Q[qc] @ self.A_t
        if self.ward_div:
            key.div_(self.inv[qc][:, None] + self.inv[None, :])
        key[torch.arange(len(qc), device=self.device), qc] = float("inf")
        return key.view(len(qc), self.S, self.w)

    def _low(self, v, err, sq, segs=None):
        """Lower bound (float32) on the exact key, up to the shift, from float32 keys v."""
        if self.ward_div:
            return v - 1e-5 * v.abs() - err * sq
        low = v - err
        if self.grouped:
            gs = self.seg_size[None, :] if segs is None else self.seg_size[segs]
            low = low * (sq * gs / (sq + gs))
        return low

    def _dev(self, x):
        return torch.from_numpy(np.asarray(x, np.float32)).to(self.device)[:, None]

    def candidates(self, query):
        """Candidate cluster ids (n, k), -1 for none, and per query a float64
        radius: every active cluster that is not a candidate has an exact key
        of at least that."""
        qcol = torch.from_numpy(self.col_of[query]).to(self.device)
        err = self._dev(self._err(query) * 1.001)
        sq = self._dev(self.size[query])
        kk, w = self.kk, self.w
        inf = float("inf")
        out, rad = [], []
        for s in range(0, len(query), self.rows):
            qc = qcol[s:s + self.rows]
            e, z = err[s:s + self.rows], sq[s:s + self.rows]
            k3 = self._keys(qc)
            q = len(qc)
            i = k3.argmin(2)                                   # best of every segment
            v = k3.gather(2, i[:, :, None])[:, :, 0]
            vals, order = self._low(v, e, z).sort(1)
            chosen = order[:, :kk]
            imin = i.gather(1, chosen)
            rows = (chosen + self.S * torch.arange(q, device=self.device)[:, None]).reshape(-1)
            sub = k3.reshape(q * self.S, w).index_select(0, rows)
            sub.scatter_(1, imin.reshape(-1, 1), inf)
            r = self._low(sub.amin(1).view(q, kk), e, z, chosen).amin(1)   # runners-up of the chosen segments
            if self.S > kk:
                r = torch.minimum(r, vals[:, kk])              # best of the segments not chosen
            out.append(chosen * w + imin)
            rad.append(r)
        cand = self.cols[torch.cat(out).cpu().numpy()]
        r = torch.cat(rad).cpu().numpy().astype(np.float64)
        with np.errstate(invalid="ignore"):
            r = np.where(r > 1e29, np.inf, r - 4e-6 * np.abs(r))   # float32 arithmetic in _low
        return cand, r + self._shift(query)

    def within(self, query, key):
        """All (query index, cluster id) pairs whose exact key could be below key (n,)."""
        err = self._err(query) * 1.001
        if self.grouped:
            sq = self.size[query][:, None]
            gs = self.seg_size64[None, :]
            thr = key[:, None] * (sq + gs) / (sq * gs) + err[:, None]
        elif self.ward_div:
            thr = (key + 1e-5 * np.abs(key) + err * self.size[query])[:, None]
        else:
            thr = (key - self._shift(query) + err)[:, None]
        thr = thr + 4e-6 * np.abs(thr)
        thr = torch.from_numpy(thr.astype(np.float32)).to(self.device)[:, :, None]
        qcol = torch.from_numpy(self.col_of[query]).to(self.device)
        qi, ci = [], []
        for s in range(0, len(query), self.rows):
            nz = (self._keys(qcol[s:s + self.rows]) <= thr[s:s + self.rows]).nonzero().cpu().numpy()
            qi.append(nz[:, 0] + s)
            ci.append(self.cols[nz[:, 1] * self.w + nz[:, 2]])
        return np.concatenate(qi), np.concatenate(ci)


def linkage(X, method="ward", device=None, k=16, segs=256, budget=1 << 25, cache=True,
            width=None, return_stats=False):
    """Exact linkage matrix in scipy's format, for the methods in METHODS.

    device: "mps", "cuda" or "cpu" (default: the GPU if there is one).
    k: candidates per cluster that come back from a full search.
    segs: about how many segments the columns are cut into for the search.
    budget: number of float32 elements per distance block on the device.
    cache: keep each cluster's neighbour list across merges and use it before
       falling back to a full search.
    width: entries kept per neighbour list (default k).
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
    W = width or k

    total = 2 * n - 1
    C = np.empty((total, d))
    C[:n] = X
    G = torch.empty((total, d), dtype=torch.float32, device=device)   # float32 copy of C on the device
    G[:n] = torch.from_numpy(X.astype(np.float32)).to(device)
    size = np.ones(total)
    var = np.zeros(total)
    norm = np.zeros(total)
    norm[:n] = np.sqrt(np.einsum("ij,ij->i", X, X))
    rep = np.arange(total)          # one member point of each cluster
    nn_id = np.full(total, -1, np.int64)
    nn_key = np.full(total, np.inf)
    valid = np.zeros(total, bool)   # nn_id is the true nearest neighbour right now
    gone = np.zeros(total, bool)
    succ = np.arange(total)         # the cluster this one was merged into
    L_id = np.full((total, W), -1, np.int64)   # neighbour list, nearest first
    L_key = np.full((total, W), np.inf)
    rad = np.full(total, -np.inf)   # every cluster outside the list is at least this far

    active = np.arange(n, dtype=np.int64)
    m_a, m_b, m_h = [], [], []
    # Identical rows merge first, at height zero, in any order. Do that up
    # front: they would otherwise be exact ties all the way through.
    probe = X @ np.random.default_rng(0).standard_normal(d)
    if len(np.unique(probe)) < n:
        _, first, inverse = np.unique(X, axis=0, return_index=True, return_inverse=True)
        first = first[inverse.ravel()]                 # first row equal to each row
        extra = np.nonzero(first != np.arange(n))[0]
        if len(extra):
            m_a.append(first[extra])
            m_b.append(extra)
            m_h.append(np.zeros(len(extra)))
            active = np.unique(first)
            size[active] = np.bincount(first, minlength=n)[active]
    fresh = active.copy()           # clusters made in the last round
    next_id = n
    stats = {"rounds": [], "t_gpu": 0.0, "t_verify": 0.0, "t_cache": 0.0, "pair_evals": 0,
             "full_searches": 0, "rechecked": 0, "fresh_hits": 0, "fresh_total": 0,
             "orphan_hits": 0, "orphan_total": 0, "tie_rounds": 0, "recheck_pairs": 0}

    def exact(q, c):
        """Float64 keys for cluster ids q (n,) against c (n, w), in chunks."""
        out = np.empty(c.shape)
        step = max(1, (1 << 25) // (c.shape[1] * d))
        for s in range(0, len(q), step):
            out[s:s + step] = _exact_key(method, C, size, var, q[s:s + step], c[s:s + step])
        return out

    def find(ids):
        """Current cluster of each id (-1 stays -1)."""
        cur = ids.copy()
        while True:
            nxt = np.where(cur < 0, -1, succ[cur])
            if np.array_equal(nxt, cur):
                return cur
            cur = nxt

    def store(q, ids, keys, r):
        """Set the neighbour lists of clusters q from candidates with exact keys.

        Outside the candidates every cluster is at least r away. Returns which
        clusters got a nearest neighbour out of it."""
        keys = np.where((ids >= 0) & (keys <= r[:, None]), keys, np.inf)
        o = np.lexsort((ids, keys), axis=1)
        ids = np.take_along_axis(ids, o, 1)
        keys = np.take_along_axis(keys, o, 1)
        dup = (ids[:, 1:] == ids[:, :-1]) & np.isfinite(keys[:, 1:])
        if dup.any():
            keys[:, 1:][dup] = np.inf
            o = np.lexsort((ids, keys), axis=1)
            ids = np.take_along_axis(ids, o, 1)
            keys = np.take_along_axis(keys, o, 1)
        w = ids.shape[1]
        if w > W:
            r = np.minimum(r, keys[:, W])        # the nearest entry that does not fit
            ids, keys = ids[:, :W], keys[:, :W]
        ids = np.where(np.isfinite(keys), ids, -1)
        L_id[q, :] = -1
        L_key[q, :] = np.inf
        L_id[q, :ids.shape[1]] = ids
        L_key[q, :ids.shape[1]] = keys
        rad[q] = r
        ok = ids[:, 0] >= 0
        nn_id[q[ok]] = ids[ok, 0]
        nn_key[q[ok]] = keys[ok, 0]
        valid[q[ok]] = True
        return ok

    def from_lists(q, ids, r, known=None):
        """Try to settle clusters q from inherited list entries `ids` (n, w)."""
        cur = find(ids)
        cur[cur == q[:, None]] = -1
        if known is None:                        # two parents' lists overlap: drop repeats
            cur = np.sort(cur, axis=1)
            cur[:, 1:][cur[:, 1:] == cur[:, :-1]] = -1
        keys = np.full(cur.shape, np.inf)
        need = cur >= 0
        if known is not None:                    # entries that did not change keep their key
            same = need & (cur == ids)
            keys[same] = known[same]
            need &= ~same
        rows, cols = np.nonzero(need)
        if len(rows):
            keys[rows, cols] = exact(q[rows], cur[rows, cols][:, None])[:, 0]
        return store(q, cur, keys, r)

    def inherit(a, b):
        """Radius for the lists a new cluster a+b inherits: a lower bound on its
        key to any cluster that holds no member of either parent's list."""
        sa, sb, ra, rb = size[a], size[b], rad[a], rad[b]
        if method != "ward":                     # the key to a+b is the size-weighted mean
            r = (sa * ra + sb * rb) / (sa + sb)
        else:
            # Lance-Williams for Ward, with x = size of the third cluster:
            # key >= (sa ra + sb rb + x (ra + rb - key(a, b))) / (sa + sb + x),
            # monotone in x, so the minimum is at x = 1 or x = infinity
            g = ra + rb - nn_key[a]
            r = np.minimum((sa * ra + sb * rb + g) / (sa + sb + 1.0), g)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(r), r - 1e-12 * np.abs(r), r)

    def full_search(searcher, query):
        t0 = time.perf_counter()
        cand, r = searcher.candidates(query)
        stats["t_gpu"] += time.perf_counter() - t0
        stats["pair_evals"] += len(query) * searcher.m
        stats["full_searches"] += len(query)
        t0 = time.perf_counter()
        cand = np.where(cand < 0, query[:, None], cand)
        keys = exact(query, cand)
        keys[cand == query[:, None]] = np.inf
        best = keys.min(1)
        sure = best <= r
        store(query[sure], cand[sure], keys[sure], r[sure])
        stats["t_verify"] += time.perf_counter() - t0
        if sure.all():
            return
        # float32 could not separate the best candidate from the other columns:
        # fetch every column that could be nearer and check it in float64
        q = query[~sure]
        t0 = time.perf_counter()
        qi, ci = searcher.within(q, best[~sure])
        stats["t_gpu"] += time.perf_counter() - t0
        stats["pair_evals"] += len(q) * searcher.m
        stats["rechecked"] += len(q)
        stats["recheck_pairs"] += len(qi)
        if not stats.get("warned") and stats["recheck_pairs"] > 50 * n + 1_000_000:
            stats["warned"] = True
            warnings.warn("many neighbours are closer together than float32 can resolve; "
                          "they are being checked in float64 on the CPU, which is slow", stacklevel=3)
        t0 = time.perf_counter()
        ek = exact(q[qi], ci[:, None])[:, 0]
        o = np.lexsort((ci, ek, qi))
        qi, ci, ek = qi[o], ci[o], ek[o]
        first = np.r_[True, qi[1:] != qi[:-1]]
        q = q[qi[first]]
        L_id[q, :] = -1
        L_key[q, :] = np.inf
        L_id[q, 0] = ci[first]
        L_key[q, 0] = ek[first]
        rad[q] = ek[first]
        nn_id[q] = ci[first]
        nn_key[q] = ek[first]
        valid[q] = True
        stats["t_verify"] += time.perf_counter() - t0

    def reciprocal():
        ok = valid[active]
        a = active[ok]
        b = nn_id[a]
        keep = valid[b] & (nn_id[b] == a) & (a < b)
        return a[keep]

    pa = pb = None
    while len(active) > 1:
        m = len(active)
        if cache and pa is not None:
            t0 = time.perf_counter()
            ok = from_lists(fresh, np.concatenate((L_id[pa], L_id[pb]), 1), inherit(pa, pb))
            stats["fresh_hits"] += int(ok.sum())
            stats["fresh_total"] += len(fresh)
            old = active[: m - len(fresh)]
            orphans = old[~valid[old]]
            ok = from_lists(orphans, L_id[orphans], rad[orphans], known=L_key[orphans])
            stats["orphan_hits"] += int(ok.sum())
            stats["orphan_total"] += len(orphans)
            stats["t_cache"] += time.perf_counter() - t0
        todo = active[~valid[active]]
        if len(todo):
            t0 = time.perf_counter()
            searcher = _Searcher(method, G, size, var, norm, active, k, segs, budget, device)
            stats["t_gpu"] += time.perf_counter() - t0
            full_search(searcher, todo)
            del searcher
        a = reciprocal()
        if len(a) == 0:
            # Only with exact ties. Any pair at the smallest key is a valid next
            # merge, and stays one after other pairs at that key have merged, so
            # take as many disjoint ones as there are.
            kmin = nn_key[active].min()
            used, xs, ys = set(), [], []
            for x in active[nn_key[active] == kmin].tolist():
                y = int(nn_id[x])
                if x not in used and y not in used:
                    used.update((x, y))
                    xs.append(min(x, y))
                    ys.append(max(x, y))
            a = np.array(xs)
            nn_id[a] = ys
            nn_key[a] = kmin
            stats["tie_rounds"] += 1
        b = nn_id[a]
        p = len(a)
        new = np.arange(next_id, next_id + p)
        next_id += p
        sa, sb = size[a], size[b]
        sn = sa + sb
        C[new] = (C[a] * sa[:, None] + C[b] * sb[:, None]) / sn[:, None]
        norm[new] = np.sqrt(np.einsum("ij,ij->i", C[new], C[new]))
        G[next_id - p:next_id] = torch.from_numpy(C[new].astype(np.float32)).to(device)
        if method == "average_sqeuclidean":
            da = np.einsum("ij,ij->i", C[a] - C[new], C[a] - C[new])
            db = np.einsum("ij,ij->i", C[b] - C[new], C[b] - C[new])
            var[new] = (sa * (var[a] + da) + sb * (var[b] + db)) / sn
        size[new] = sn
        rep[new] = rep[a]
        succ[a] = new
        succ[b] = new
        m_a.append(rep[a])
        m_b.append(rep[b])
        m_h.append(_height(method, nn_key[a]))
        gone[a] = True
        gone[b] = True
        keep = active[~gone[active]]
        valid[keep[gone[nn_id[keep]]]] = False   # their nearest neighbour was merged away
        stats["rounds"].append((m, len(todo), p))
        active = np.concatenate((keep, new))
        fresh, pa, pb = new, a, b

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
