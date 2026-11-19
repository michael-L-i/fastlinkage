"""Turn the list of merges into scipy's linkage matrix."""
import numpy as np


def to_scipy(ra, rb, h, n):
    """Linkage matrix in scipy's format from merges given in any order.

    ra, rb: one member point of each of the two clusters merged; h: the merge
    heights; n: number of points. Merges are sorted by height and the clusters
    numbered the way scipy does."""
    order = np.argsort(h, kind="stable")
    parent = list(range(n))             # plain lists: the loop below is scalar work
    label = list(range(n))
    csize = [1] * n
    left, right, sizes = [0] * (n - 1), [0] * (n - 1), [0] * (n - 1)
    ra = ra.tolist()
    rb = rb.tolist()
    for i, j in enumerate(order.tolist()):
        x = ra[j]
        while parent[x] != x:           # find the root, halving the path on the way
            parent[x] = x = parent[parent[x]]
        y = rb[j]
        while parent[y] != y:
            parent[y] = y = parent[parent[y]]
        lx, ly = label[x], label[y]
        left[i], right[i] = (lx, ly) if lx < ly else (ly, lx)
        sizes[i] = csize[x] = csize[x] + csize[y]
        parent[y] = x
        label[x] = n + i
    return np.column_stack((left, right, h[order], sizes)).astype(np.float64)
