"""Exactness tests against scipy and scikit-learn.

They run on the CPU and, when the machine has one, on the GPU (see conftest.py)."""
import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform

from fastlinkage import METHODS, AgglomerativeClustering, cut, linkage

SCIPY = {"ward": ("ward", "euclidean"), "average_cosine": ("average", "cosine"),
         "average_sqeuclidean": ("average", "sqeuclidean")}


def blobs(n, d, seed=0, centres=12):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(centres, d)) * 4
    return c[rng.integers(0, centres, n)] + rng.normal(size=(n, d))


def same_partition(a, b):
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n,d", [(2, 3), (3, 5), (50, 4), (1500, 16)])
def test_same_tree_as_scipy(method, n, d, device):
    X = blobs(n, d)
    Z = linkage(X, method, device=device)
    Zs = scipy_linkage(X, *SCIPY[method])
    assert Z.shape == Zs.shape
    np.testing.assert_allclose(Z[:, 2], Zs[:, 2], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(np.sort(Z[:, 3]), np.sort(Zs[:, 3]))
    for k in (2, 5, 20):
        if k < n:
            assert same_partition(fcluster(Z, k, "maxclust"), fcluster(Zs, k, "maxclust"))


@pytest.mark.parametrize("method", METHODS)
def test_cache_and_brute_force_agree(method, device):
    X = blobs(2000, 8, seed=1)
    a = linkage(X, method, device=device, cache=True)
    b = linkage(X, method, device=device, cache=False)
    np.testing.assert_allclose(a[:, 2], b[:, 2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_duplicate_rows(method, device):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 6))[rng.integers(0, 200, 900)]
    Z = linkage(X, method, device=device)
    Zs = scipy_linkage(X, *SCIPY[method])
    np.testing.assert_allclose(Z[:, 2], Zs[:, 2], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_near_duplicates_below_float32_resolution(method, device):
    rng = np.random.default_rng(3)
    X = np.repeat(rng.normal(size=(100, 16)), 6, axis=0) + 1e-7 * rng.normal(size=(600, 16))
    Z = linkage(X, method, device=device)
    Zs = scipy_linkage(X, *SCIPY[method])
    np.testing.assert_allclose(Z[:, 2], Zs[:, 2], rtol=1e-7, atol=1e-13)


def first_wrong_merge(X, method, Z, tol=1e-9):
    """Replay Z on the full distance matrix (Lance-Williams update). Returns the
    first merge that is not a closest pair at its step or has the wrong height,
    or None. With tied distances several trees are correct; this accepts all of
    them, where a comparison with scipy's heights would not."""
    n = len(X)
    D = np.full((2 * n - 1, 2 * n - 1), np.inf)
    D[:n, :n] = squareform(pdist(X, SCIPY[method][1]))
    np.fill_diagonal(D, np.inf)
    size = np.ones(2 * n - 1)
    alive = np.arange(2 * n - 1) < n
    for i, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        closest = D[np.ix_(alive, alive)].min()
        slack = tol * max(closest, h) + 1e-13       # 1e-13: pdist's own rounding near zero
        if not (alive[a] and alive[b]) or abs(D[a, b] - h) > slack or D[a, b] - closest > slack:
            return i
        if method == "ward":
            new = np.sqrt(np.maximum(((size[a] + size) * D[a] ** 2 + (size[b] + size) * D[b] ** 2
                                      - size * h ** 2) / (size[a] + size[b] + size), 0))
        else:
            new = (size[a] * D[a] + size[b] * D[b]) / (size[a] + size[b])
        alive[[a, b]] = False
        new[~alive] = np.inf
        D[[a, b]] = D[:, [a, b]] = np.inf
        D[n + i] = D[:, n + i] = new
        alive[n + i], size[n + i] = True, size[a] + size[b]
    return None


def tied_data():
    rng = np.random.default_rng(8)
    grid = np.stack(np.meshgrid(np.arange(18.0), np.arange(18.0)), -1).reshape(-1, 2) + 1
    return {"integers on a line": rng.integers(0, 40, (400, 1)) + 1.0,
            "evenly spaced line": np.arange(300.0)[:, None] + 1,
            "grid": grid,
            "binary": rng.integers(0, 2, (400, 10)) + 0.5}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", list(tied_data()))
def test_tied_distances_give_a_correct_tree(method, name, device):
    X = tied_data()[name]
    if method == "average_cosine" and X.shape[1] == 1:
        pytest.skip("cosine distance needs two features")
    assert first_wrong_merge(X, method, linkage(X, method, device=device)) is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [1e-150, 1e-15, 1e15, 1e150])
def test_any_units(method, scale, device):
    if method == "average_sqeuclidean" and scale == 1e150:
        pytest.skip("the squared distances themselves overflow float64")
    X = blobs(400, 8, seed=9) * scale
    Z = linkage(X, method, device=device)
    Zs = scipy_linkage(X, *SCIPY[method])
    np.testing.assert_allclose(Z[:, 2], Zs[:, 2], rtol=1e-9)


def test_cut(device):
    X = blobs(800, 8, seed=4)
    Z = linkage(X, "ward", device=device)
    labels = cut(Z, n_clusters=7)
    assert labels.max() == 6 and same_partition(labels, fcluster(Z, 7, "maxclust"))
    t = float(Z[-20, 2])
    labels = cut(Z, distance_threshold=t)          # merges at or above t are not applied
    assert labels.max() + 1 == 800 - int(np.sum(Z[:, 2] < t))
    assert cut(Z, n_clusters=1).max() == 0 and cut(Z, n_clusters=800).max() == 799
    with pytest.raises(ValueError):
        cut(Z)


def test_estimator_matches_sklearn(device):
    sk = pytest.importorskip("sklearn.cluster")
    X = blobs(600, 10, seed=5)
    for kw in (dict(n_clusters=6, linkage="ward"),
               dict(n_clusters=None, distance_threshold=0.5, linkage="average", metric="cosine"),
               dict(n_clusters=9, linkage="average", metric="cosine")):
        ours = AgglomerativeClustering(device=device, **kw).fit(X)
        ref = sk.AgglomerativeClustering(**kw).fit(X)
        assert ours.n_clusters_ == ref.n_clusters_
        assert same_partition(ours.labels_, ref.labels_)
        assert ours.children_.shape == ref.children_.shape


def test_scipy_signature(device):
    X = blobs(300, 6, seed=6)
    for args in (("ward",), ("ward", "euclidean"), ("average", "cosine"), ("average", "sqeuclidean")):
        np.testing.assert_allclose(linkage(X, *args, device=device)[:, 2], scipy_linkage(X, *args)[:, 2],
                                   rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(linkage(X, "average_cosine", device=device),
                                  linkage(X, "average", metric="cosine", device=device))


def test_input_types(device):
    import torch

    X = blobs(200, 5, seed=7).astype(np.float32)
    Z = linkage(X, device=device)
    np.testing.assert_array_equal(Z, linkage(torch.from_numpy(X), device=device))
    np.testing.assert_array_equal(Z, linkage(X.tolist(), device=device))
    np.testing.assert_allclose(Z[:, 2], scipy_linkage(X.astype(np.float64), "ward")[:, 2], rtol=1e-9)


def test_estimator_params():
    base = pytest.importorskip("sklearn.base")
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, linkage="average", metric="cosine")
    copy = base.clone(model)
    assert copy is not model and copy.get_params() == model.get_params()
    assert copy.set_params(n_clusters=4, distance_threshold=None).n_clusters == 4
    assert "distance_threshold=None" in repr(copy)
    with pytest.raises(ValueError):
        copy.set_params(nonsense=1)
    X = blobs(100, 7)
    assert copy.fit(X).n_features_in_ == 7 and len(copy.labels_) == 100


def test_bad_arguments():
    X = blobs(20, 3)
    with pytest.raises(ValueError):
        linkage(X, "complete")
    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="complete").fit(X)
    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=3, distance_threshold=1.0).fit(X)
    with pytest.raises(ValueError):
        linkage(np.zeros((5, 3)), "average_cosine")
    with pytest.raises(ValueError, match="metric"):
        linkage(X, "average")                       # scipy's default metric needs the matrix
    with pytest.raises(ValueError):
        linkage(X, "ward", "cosine")
    with pytest.raises(ValueError, match="2-D"):
        linkage(X[0])
    with pytest.raises(ValueError, match="at least 2 rows"):
        linkage(X[:1])
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        linkage(bad)
