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
