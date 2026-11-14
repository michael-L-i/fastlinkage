"""Flat clusters from a linkage matrix, and a scikit-learn style estimator."""
import numpy as np

from .core import linkage

_METHOD = {("ward", "euclidean"): "ward", ("average", "cosine"): "average_cosine",
           ("average", "sqeuclidean"): "average_sqeuclidean"}


def cut(Z, n_clusters=None, distance_threshold=None):
    """Cluster label (0 .. k-1) of every point after cutting the tree.

    Give exactly one of:
      n_clusters: stop when this many clusters are left.
      distance_threshold: apply every merge whose height is below it (the
        scikit-learn rule: clusters at or above the threshold are not merged).
    """
    if (n_clusters is None) == (distance_threshold is None):
        raise ValueError("give exactly one of n_clusters and distance_threshold")
    Z = np.asarray(Z)
    n = len(Z) + 1
    if n_clusters is not None:
        if not 1 <= n_clusters <= n:
            raise ValueError(f"n_clusters must be between 1 and {n}")
        k = n - int(n_clusters)
    else:
        k = int(np.searchsorted(Z[:, 2], distance_threshold, side="left"))
    root = np.arange(n + k)
    root[Z[:k, 0].astype(np.int64)] = n + np.arange(k)
    root[Z[:k, 1].astype(np.int64)] = n + np.arange(k)
    while True:                       # pointer jumping: follow parents to the top
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    return np.unique(root[:n], return_inverse=True)[1]


class AgglomerativeClustering:
    """Exact agglomerative clustering with scikit-learn's interface.

    Supported: linkage="ward" (metric="euclidean"), and linkage="average" with
    metric="cosine" or "sqeuclidean". Set n_clusters, or set n_clusters=None
    and distance_threshold.

    After fit: labels_, n_clusters_, n_leaves_, children_, distances_, and
    linkage_matrix_ (scipy's format, for scipy.cluster.hierarchy.dendrogram).
    """

    def __init__(self, n_clusters=2, *, metric="euclidean", linkage="ward",
                 distance_threshold=None, device=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.device = device

    def fit(self, X, y=None):
        method = _METHOD.get((self.linkage, self.metric))
        if method is None:
            raise ValueError(
                f"linkage={self.linkage!r} with metric={self.metric!r} is not supported; "
                f"supported pairs: {sorted(_METHOD)}")
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError("exactly one of n_clusters and distance_threshold has to be set, "
                             "and the other needs to be None")
        Z = linkage(X, method=method, device=self.device)
        self.linkage_matrix_ = Z
        self.children_ = Z[:, :2].astype(np.int64)
        self.distances_ = Z[:, 2]
        self.n_leaves_ = len(Z) + 1
        self.labels_ = cut(Z, self.n_clusters, self.distance_threshold)
        self.n_clusters_ = int(self.labels_.max()) + 1
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
