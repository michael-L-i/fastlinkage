"""Nearest-neighbour search among the active clusters, in float32 on the device."""
import numpy as np
import torch

MAX_SIZE_GROUPS = 24        # Ward: sort columns by cluster size while there are this few sizes
FLOAT32_ROUNDOFF = 6e-8     # float32 unit roundoff


class Searcher:
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

    def __init__(self, method, means, size, var, norm, active, k, segs, budget, device):
        """means(ids, device): float32 means of those cluster ids, on the device."""
        self.method = method
        self.device = device
        self.size, self.var, self.norm = size, var, norm
        m = len(active)
        self.m = m
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
            for st, c in zip(start, counts, strict=True):
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
        self.budget = budget
        self.rows = max(1, budget // len(cols))
        real = cols >= 0
        Cg = means(np.maximum(cols, 0), device)
        d = Cg.shape[1]
        if not real.all():
            Cg = Cg * torch.from_numpy(real.astype(np.float32)).to(device)[:, None]
        pad = torch.from_numpy(np.where(real, 0.0, self.BIG).astype(np.float32)).to(device)[:, None]
        one = torch.ones_like(pad)
        self.pad, self.vp = pad, None
        if method == "average_cosine":
            self.A_t = torch.cat((-Cg, pad), 1).T.contiguous()     # key = -dot
            self.Q = torch.cat((Cg, one), 1)
        else:
            n2 = (Cg * Cg).sum(1, keepdim=True)
            extra = n2 + pad
            if method == "average_sqeuclidean":
                vp = np.zeros(len(cols), np.float32)
                vp[real] = var[cols[real]]
                self.vp = torch.from_numpy(vp).to(device)[:, None]
                extra = extra + self.vp
            self.A_t = torch.cat((-2.0 * Cg, one, extra), 1).T.contiguous()
            self.Q = torch.cat((Cg, n2, one), 1)  # Q @ A_t = squared distance (+ column variance)
        if self.ward_div:
            sp = np.ones(len(cols), np.float32)
            sp[real] = s_act
            self.inv = 1.0 / torch.from_numpy(sp).to(device)
        self.col_of = np.full(len(size), -1, np.int64)
        self.col_of[cols[real]] = np.nonzero(real)[0]
        # float32 rounding bound on a key, relative to the sum of the absolute terms
        self.gamma = 4.0 * (d + 4) * FLOAT32_ROUNDOFF
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
        return self._finish(torch.cat(out), torch.cat(rad), self._shift(query))

    def _finish(self, col, rad, shift):
        """Candidate ids and float64 radii from device columns and float32 bounds."""
        cand = self.cols[col.cpu().numpy()]
        r = rad.cpu().numpy().astype(np.float64)
        with np.errstate(invalid="ignore"):
            r = np.where(r > 1e29, np.inf, r - 4e-6 * np.abs(r))   # float32 arithmetic in _low
        return cand, r + shift

    # ---- every cluster at once ----

    def symmetric_fits(self):
        """Whether candidates_all can hold its table of segment minima."""
        return len(self.cols) * self.S <= 8 * self.budget

    def candidates_all(self):
        """candidates() for every active cluster at once, at about half the cost.

        The key is symmetric, so one block of keys serves its rows and its
        columns: only pairs (i, j) with i < j are computed, and each block is
        reduced along both axes into a table of segment minima, one row per
        cluster. A second look at the k chosen segments of every cluster, one
        segment at a time, finds the nearest column and the runner-up.

        Returns the cluster ids in column order, then what candidates() returns."""
        dev, w, S, kk = self.device, self.w, self.S, self.kk
        inf = float("inf")
        ncols = len(self.cols)
        real = self.cols >= 0
        # the same key from either side: padding and (for average linkage with
        # squared distances) the variance go into the rows as well
        if self.method == "average_cosine":
            Q = torch.cat((self.Q, self.pad), 1)
            A_t = torch.cat((self.A_t, torch.ones((1, ncols), device=dev)), 0)
        else:
            Q = self.Q.clone()
            Q[:, -2:-1] += self.pad if self.vp is None else self.pad + self.vp
            A_t = self.A_t

        V = torch.full((ncols, S), inf, device=dev)      # V[j, s]: smallest key from j into segment s
        ar = torch.arange(ncols, device=dev)
        r0 = 0
        while r0 < ncols:
            a = r0 // w
            c0 = a * w                                   # columns start at the segment boundary
            C = ncols - c0
            R = max(1, self.budget // C)
            if r0 == c0 and R >= w:
                R = min(R // w * w, ncols - r0)          # whole segments
            else:
                R = min(R, c0 + w - r0)                  # the rest of one segment
            r1 = r0 + R
            b = (r1 - 1) // w + 1
            K = Q[r0:r1] @ A_t[:, c0:]
            if self.ward_div:
                K.div_(self.inv[r0:r1, None] + self.inv[None, c0:])
            K[:, :r1 - c0].masked_fill_(ar[c0:r1][None, :] <= ar[r0:r1][:, None], inf)   # keep i < j
            V[r0:r1, a:] = torch.minimum(V[r0:r1, a:], K.view(R, S - a, w).amin(2))
            V[c0:, a:b] = torch.minimum(V[c0:, a:b], K.view(b - a, R // (b - a), C).amin(1).T)
            r0 = r1
            del K

        qcol = torch.from_numpy(np.nonzero(real)[0]).to(dev)
        query = self.cols[real]
        m = len(query)
        err = self._dev(self._err(query) * 1.001)
        sq = self._dev(self.size[query])
        chosen = torch.empty((m, kk), dtype=torch.int64, device=dev)
        rest = torch.full((m,), inf, device=dev)
        step = max(1, self.budget // S)
        for s in range(0, m, step):
            vals, order = self._low(V[qcol[s:s + step]], err[s:s + step], sq[s:s + step]).sort(1)
            chosen[s:s + step] = order[:, :kk]
            if S > kk:
                rest[s:s + step] = vals[:, kk]           # best of the segments not chosen
        del V

        # second look: every cluster against each segment it chose. The
        # (cluster, segment) entries are sorted by segment and segments with a
        # similar number of entries are stacked into one batched product.
        flat = chosen.reshape(-1)
        by_seg = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=S).cpu().numpy()
        start = np.cumsum(counts) - counts
        A3 = A_t.view(-1, S, w).permute(1, 0, 2).contiguous()        # (segment, feature, column)
        inv3 = self.inv.view(S, 1, w) if self.ward_div else None
        imin = torch.empty(m * kk, dtype=torch.int64, device=dev)
        runner = torch.empty(m * kk, device=dev)
        todo = [int(x) for x in np.argsort(-counts, kind="stable") if counts[x]]
        while todo:
            height = int(counts[todo[0]])                            # most entries in this batch
            group = todo[:max(1, self.budget // (height * w))]
            todo = todo[len(group):]
            G = len(group)
            entry = torch.cat([by_seg[start[g]:start[g] + counts[g]] for g in group])
            slot = np.concatenate([i * height + np.arange(counts[g]) for i, g in enumerate(group)])
            slot = torch.from_numpy(slot).to(dev)
            qc = torch.zeros(G * height, dtype=torch.int64, device=dev)   # unused slots repeat column 0
            qc[slot] = qcol[entry // kk]
            segs = torch.tensor(group, device=dev)
            K = torch.bmm(Q[qc].view(G, height, -1), A3[segs])       # (segment, entry, column)
            if self.ward_div:
                K.div_(self.inv[qc].view(G, height, 1) + inv3[segs])
            K = K.view(G * height, w)
            own = qc - segs.repeat_interleave(height) * w            # the cluster's own column, if in this segment
            inside = ((own >= 0) & (own < w))[:, None]
            own = own.clamp(0, w - 1)[:, None]
            K.scatter_(1, own, torch.where(inside, inf, K.gather(1, own)))
            i = K.argmin(1, keepdim=True)
            imin[entry] = i[slot, 0]
            K.scatter_(1, i, inf)
            runner[entry] = K.amin(1)[slot]
            del K

        r = self._low(runner.view(m, kk), err, sq, chosen).amin(1)
        r = torch.minimum(r, rest)
        shift = np.ones(m) if self.method == "average_cosine" else np.zeros(m)
        return (query, *self._finish(chosen * w + imin.view(m, kk), r, shift))

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
