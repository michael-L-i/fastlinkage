# Changelog

## Unreleased

- Comparison with cuhc on the same GPU, in the README.

## 0.2.1 (2026-11-18)

- Fixed: Ward and squared-Euclidean average linkage returned a wrong tree
  when the values were very large (around 1e15 and above). The input is now
  rescaled by a power of two before the search, so the units do not matter.
- Fixed: cosine linkage overflowed on rows with very large or very small
  values.
- Tests for tied distances (grids, integers, binary data) and for inputs
  from 1e-150 to 1e150.

## 0.2.0 (2026-11-17)

- `linkage` takes SciPy's arguments: `linkage(X, "average", metric="cosine")`.
  The one-word names (`"average_cosine"`) still work.
- Changed: every argument after `metric` is keyword-only, so `device` must be
  passed as `device=...`.
- Torch tensors are accepted as input.
- Clear errors for input that is not 2-D, has fewer than two rows, or
  contains NaN or infinity, and for unsupported linkages.
- `AgglomerativeClustering` has `get_params`, `set_params` and
  `n_features_in_`, so `sklearn.base.clone` and pipelines accept it.
- Type hints and a `py.typed` marker.

## 0.1.1 (2026-11-16)

- Verified on an NVIDIA GPU (Tesla T4): same trees as SciPy and fastcluster.
- On CUDA the float64 checks run on the GPU: 100,000 x 384 points went from
  31-49 s to 11-15 s on a T4 with 2 CPU cores.
- Tests run on every available device (CPU, CUDA, MPS).
- CI covers Linux, macOS and Windows.
- Colab notebook that runs the CUDA check.
- First release on PyPI: `pip install fastlinkage`.

## 0.1.0 (2026-11-15)

First release.

- `linkage`: exact Ward, average-cosine and average-squared-Euclidean trees in
  SciPy's format, without the n x n distance matrix.
- `AgglomerativeClustering`: scikit-learn style estimator on top of it.
- `cut`: flat clusters from a tree by cluster count or distance threshold.
- Runs on Apple GPUs (PyTorch MPS) and on the CPU. CUDA is untested.
