# Changelog

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
