"""Cluster summaries, nearest neighbours and the neighbour lists that survive merges.

Cluster ids: the n points are 0 .. n-1 and every merge makes the next id, so
there are at most 2n - 1. All arrays here are indexed by cluster id.
"""
import time
import warnings

import numpy as np
import torch

from ._distance import exact_key


def _float64_fits(device, nbytes):
    """Whether a float64 array of nbytes can live on the device next to the search."""
    if torch.device(device).type != "cuda":         # MPS has no float64
        return False
    free, _ = torch.cuda.mem_get_info(device)
    return nbytes <= free // 4


class Clusters:
    """State of a run: one summary, one nearest neighbour and one neighbour
    list per cluster.

    The neighbour list of cluster c is up to `width` clusters with their exact
    keys, nearest first, and a radius: every cluster that holds no member of
    the list has a key to c of at least the radius. That stays true after any
    merges (the linkages are reducible), so when c's nearest neighbour is
    merged away the next one can usually be read off the list.
    """

    def __init__(self, X, method, device, width):
        n, d = X.shape
        total = 2 * n - 1
        self.method, self.device, self.width, self.n = method, device, width, n
        self.C = np.empty((total, d))               # cluster means, float64
        self.C[:n] = X
        self.G = torch.empty((total, d), dtype=torch.float32, device=device)   # float32 copy on the device
        self.G[:n] = torch.from_numpy(X.astype(np.float32)).to(device)
        # float64 means for the exact keys: on the device when it supports
        # float64 and has room (CUDA), otherwise the numpy array itself
        self.on_device64 = _float64_fits(device, total * d * 8)
        if self.on_device64:
            self.C64 = torch.empty((total, d), dtype=torch.float64, device=device)
            self.C64[:n] = torch.from_numpy(X).to(device)
        else:
            self.C64 = torch.from_numpy(self.C)     # shares memory; threaded on the CPU
        self.size = np.ones(total)
        self.var = np.zeros(total)                  # mean squared distance to the cluster mean
        self.norm = np.zeros(total)
        self.norm[:n] = np.sqrt(np.einsum("ij,ij->i", X, X))
        self.rep = np.arange(total)                 # one member point of each cluster
        self.succ = np.arange(total)                # the cluster this one was merged into
        self.gone = np.zeros(total, bool)
        self.nn_id = np.full(total, -1, np.int64)
        self.nn_key = np.full(total, np.inf)
        self.valid = np.zeros(total, bool)          # nn_id is the true nearest neighbour right now
        self.L_id = np.full((total, width), -1, np.int64)   # neighbour list, nearest first
        self.L_key = np.full((total, width), np.inf)
        self.rad = np.full(total, -np.inf)          # every cluster outside the list is at least this far
        self.next_id = n
        self.stats = {"rounds": [], "t_gpu": 0.0, "t_verify": 0.0, "t_cache": 0.0, "pair_evals": 0,
                      "full_searches": 0, "rechecked": 0, "fresh_hits": 0, "fresh_total": 0,
                      "orphan_hits": 0, "orphan_total": 0, "tie_rounds": 0, "recheck_pairs": 0}

    # ---- exact keys and current cluster ids ----

    def exact(self, q, c):
        """Float64 keys for cluster ids q (n,) against c (n, w), in chunks."""
        out = np.empty(c.shape)
        step = max(1, (1 << 25) // (c.shape[1] * self.C.shape[1]))
        for s in range(0, len(q), step):
            out[s:s + step] = exact_key(self.method, self.C64, self.size, self.var, q[s:s + step], c[s:s + step])
        return out

    def find(self, ids):
        """Current cluster of each id (-1 stays -1)."""
        cur = ids.copy()
        while True:
            nxt = np.where(cur < 0, -1, self.succ[cur])
            if np.array_equal(nxt, cur):
                return cur
            cur = nxt

    # ---- neighbour lists ----

    def store(self, q, ids, keys, r):
        """Set the neighbour lists of clusters q from candidates with exact keys.

        Outside the candidates every cluster is at least r away. Returns which
        clusters got a nearest neighbour out of it."""
        W = self.width
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
        if ids.shape[1] > W:
            r = np.minimum(r, keys[:, W])        # the nearest entry that does not fit
            ids, keys = ids[:, :W], keys[:, :W]
        ids = np.where(np.isfinite(keys), ids, -1)
        self.L_id[q, :] = -1
        self.L_key[q, :] = np.inf
        self.L_id[q, :ids.shape[1]] = ids
        self.L_key[q, :ids.shape[1]] = keys
        self.rad[q] = r
        ok = ids[:, 0] >= 0
        self.nn_id[q[ok]] = ids[ok, 0]
        self.nn_key[q[ok]] = keys[ok, 0]
        self.valid[q[ok]] = True
        return ok

    def from_lists(self, q, ids, r, known=None):
        """Try to settle clusters q from inherited list entries `ids` (n, w).

        known: the keys of those entries, if q are old clusters (entries that
        were not merged keep their key). Returns which clusters were settled."""
        cur = self.find(ids)
        cur[cur == q[:, None]] = -1
        if known is None:                        # two parents' lists overlap: drop repeats
            cur = np.sort(cur, axis=1)
            cur[:, 1:][cur[:, 1:] == cur[:, :-1]] = -1
        keys = np.full(cur.shape, np.inf)
        need = cur >= 0
        if known is not None:
            same = need & (cur == ids)
            keys[same] = known[same]
            need &= ~same
        rows, cols = np.nonzero(need)
        if len(rows):
            keys[rows, cols] = self.exact(q[rows], cur[rows, cols][:, None])[:, 0]
        return self.store(q, cur, keys, r)

    def inherited_radius(self, a, b):
        """Radius for the lists a new cluster a+b inherits: a lower bound on its
        key to any cluster that holds no member of either parent's list."""
        sa, sb, ra, rb = self.size[a], self.size[b], self.rad[a], self.rad[b]
        if self.method != "ward":                # the key to a+b is the size-weighted mean
            r = (sa * ra + sb * rb) / (sa + sb)
        else:
            # Lance-Williams for Ward, with x = size of the third cluster:
            # key >= (sa ra + sb rb + x (ra + rb - key(a, b))) / (sa + sb + x),
            # monotone in x, so the minimum is at x = 1 or x = infinity
            g = ra + rb - self.nn_key[a]
            r = np.minimum((sa * ra + sb * rb + g) / (sa + sb + 1.0), g)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(r), r - 1e-12 * np.abs(r), r)

    def refresh_from_lists(self, active, fresh, pa, pb):
        """Before a search: settle the new clusters `fresh` (parents pa, pb) and
        the old clusters whose neighbour was merged away, from their lists."""
        stats = self.stats
        t0 = time.perf_counter()
        ids = np.concatenate((self.L_id[pa], self.L_id[pb]), 1)
        ok = self.from_lists(fresh, ids, self.inherited_radius(pa, pb))
        stats["fresh_hits"] += int(ok.sum())
        stats["fresh_total"] += len(fresh)
        old = active[: len(active) - len(fresh)]
        orphans = old[~self.valid[old]]
        ok = self.from_lists(orphans, self.L_id[orphans], self.rad[orphans], known=self.L_key[orphans])
        stats["orphan_hits"] += int(ok.sum())
        stats["orphan_total"] += len(orphans)
        stats["t_cache"] += time.perf_counter() - t0

    # ---- full search ----

    def full_search(self, searcher, query):
        """Nearest neighbour and neighbour list of every cluster in query, from
        a float32 search on the device checked in float64."""
        stats = self.stats
        t0 = time.perf_counter()
        cand, r = searcher.candidates(query)
        stats["t_gpu"] += time.perf_counter() - t0
        stats["pair_evals"] += len(query) * searcher.m
        stats["full_searches"] += len(query)
        t0 = time.perf_counter()
        cand = np.where(cand < 0, query[:, None], cand)
        keys = self.exact(query, cand)
        keys[cand == query[:, None]] = np.inf
        best = keys.min(1)
        sure = best <= r
        self.store(query[sure], cand[sure], keys[sure], r[sure])
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
        if not stats.get("warned") and stats["recheck_pairs"] > 50 * self.n + 1_000_000:
            stats["warned"] = True
            warnings.warn("many neighbours are closer together than float32 can resolve; "
                          "they are being checked in float64 on the CPU, which is slow", stacklevel=3)
        t0 = time.perf_counter()
        ek = self.exact(q[qi], ci[:, None])[:, 0]
        o = np.lexsort((ci, ek, qi))
        qi, ci, ek = qi[o], ci[o], ek[o]
        first = np.r_[True, qi[1:] != qi[:-1]]
        q = q[qi[first]]
        self.L_id[q, :] = -1
        self.L_key[q, :] = np.inf
        self.L_id[q, 0] = ci[first]
        self.L_key[q, 0] = ek[first]
        self.rad[q] = ek[first]
        self.nn_id[q] = ci[first]
        self.nn_key[q] = ek[first]
        self.valid[q] = True
        stats["t_verify"] += time.perf_counter() - t0

    # ---- choosing and applying merges ----

    def reciprocal_pairs(self, active):
        """The smaller id of every pair of clusters that are each other's nearest neighbour."""
        a = active[self.valid[active]]
        b = self.nn_id[a]
        keep = self.valid[b] & (self.nn_id[b] == a) & (a < b)
        return a[keep]

    def tied_pairs(self, active):
        """Fallback when no pair is reciprocal, which happens only with exact ties.

        Any pair at the smallest key is a valid next merge, and stays one after
        other pairs at that key have merged, so take as many disjoint ones as
        there are."""
        kmin = self.nn_key[active].min()
        used, xs, ys = set(), [], []
        for x in active[self.nn_key[active] == kmin].tolist():
            y = int(self.nn_id[x])
            if x not in used and y not in used:
                used.update((x, y))
                xs.append(min(x, y))
                ys.append(max(x, y))
        a = np.array(xs)
        self.nn_id[a] = ys
        self.nn_key[a] = kmin
        self.stats["tie_rounds"] += 1
        return a

    def merge(self, a, b):
        """Merge clusters a[i] and b[i] for every i; returns the new cluster ids."""
        C, size, var = self.C, self.size, self.var
        p = len(a)
        new = np.arange(self.next_id, self.next_id + p)
        self.next_id += p
        sa, sb = size[a], size[b]
        sn = sa + sb
        C[new] = (C[a] * sa[:, None] + C[b] * sb[:, None]) / sn[:, None]
        self.norm[new] = np.sqrt(np.einsum("ij,ij->i", C[new], C[new]))
        self.G[new[0]:new[0] + p] = torch.from_numpy(C[new].astype(np.float32)).to(self.device)
        if self.on_device64:
            self.C64[new[0]:new[0] + p] = torch.from_numpy(C[new]).to(self.device)
        if self.method == "average_sqeuclidean":
            da = np.einsum("ij,ij->i", C[a] - C[new], C[a] - C[new])
            db = np.einsum("ij,ij->i", C[b] - C[new], C[b] - C[new])
            var[new] = (sa * (var[a] + da) + sb * (var[b] + db)) / sn
        size[new] = sn
        self.rep[new] = self.rep[a]
        self.succ[a] = new
        self.succ[b] = new
        self.gone[a] = True
        self.gone[b] = True
        return new
