"""Turn the list of merges into scipy's linkage matrix."""
import numpy as np


def to_scipy(ra, rb, h, n):
    """Linkage matrix in scipy's format from merges given in any order.

    ra, rb: one member point of each of the two clusters merged; h: the merge
    heights; n: number of points. Merges are sorted by height and the clusters
    numbered the way scipy does."""
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
