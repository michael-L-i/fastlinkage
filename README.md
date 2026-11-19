# fastlinkage

[![tests](https://github.com/michael-L-i/fastlinkage/actions/workflows/test.yml/badge.svg)](https://github.com/michael-L-i/fastlinkage/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fastlinkage.svg)](https://pypi.org/project/fastlinkage/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Exact hierarchical clustering on the GPU, without the n x n distance matrix.

It returns the same tree as `scipy.cluster.hierarchy.linkage` for Ward and
average linkage, runs in seconds at 100,000+ points, and uses memory that
grows with the number of points, not its square.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/michael-L-i/fastlinkage/main/assets/speed-dark.svg">
  <img src="https://raw.githubusercontent.com/michael-L-i/fastlinkage/main/assets/speed-light.svg" alt="Time to build the exact tree. 20,000 text embeddings, average linkage with cosine distance: 0.8 s against 47 s for SciPy. 100,000 points with 32 features, Ward: 2.2 s against 2.3 minutes for fastcluster. 120,000 text embeddings, Ward: 8.7 s against 59 minutes for fastcluster." width="800">
</picture>

> **Status: early release.** Tested on NVIDIA GPUs (CUDA), Apple GPUs (PyTorch
> MPS) and the CPU on Linux, macOS and Windows.

## Why

SciPy and scikit-learn build the full distance matrix. At 120,000 points that
is 58 GB, so most people give up on hierarchical clustering long before that
and switch to k-means or HDBSCAN. This package keeps the exact tree and drops
the matrix:

| | SciPy / scikit-learn | fastlinkage |
|---|---|---|
| Memory | n x n distances | one mean vector per cluster |
| 20,000 text embeddings (384 features), average cosine | 47 s | 0.8 s |
| 120,000 text embeddings, average cosine | needs a 58 GB matrix | 6.9 s |
| Result | exact | exact, same tree |

## Install

```
pip install fastlinkage
```

Needs Python 3.10+, NumPy, SciPy and PyTorch. It uses CUDA or Apple MPS when
PyTorch sees one, and the CPU otherwise.

| Hardware | Status |
|---|---|
| Apple GPU (MPS) | tested; main benchmarks below |
| CPU on Linux, macOS, Windows | tested on every commit |
| NVIDIA GPU (CUDA) | tested on a Tesla T4 ([notebook](https://colab.research.google.com/github/michael-L-i/fastlinkage/blob/main/notebooks/cuda_check.ipynb)) |

The package is pure Python on top of PyTorch, so there is nothing to compile.

## Use

Drop-in for SciPy's `linkage`, with the same arguments:

```python
from fastlinkage import linkage    # was: from scipy.cluster.hierarchy import linkage

Z = linkage(X, "ward")
Z = linkage(X, "average", metric="cosine")
Z = linkage(X, "average", metric="sqeuclidean")
```

`X` is a NumPy array or a torch tensor with one point per row. Other
linkages raise an error that says so; nothing falls back silently to the
slow path.

`Z` is SciPy's linkage matrix, so everything downstream keeps working:

```python
from scipy.cluster.hierarchy import dendrogram, fcluster

labels = fcluster(Z, t=50, criterion="maxclust")
dendrogram(Z, truncate_mode="lastp", p=40)
```

Cluster embeddings without choosing the number of clusters:

```python
from fastlinkage import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0.35, linkage="average", metric="cosine"
).fit(embeddings)

model.labels_       # cluster of every row
model.n_clusters_   # how many clusters the threshold gave
```

The class follows scikit-learn's `AgglomerativeClustering` (`labels_`,
`children_`, `distances_`, `fit_predict`, `get_params`), so it works with
`sklearn.base.clone` and in pipelines. Cutting one tree at several levels
costs almost nothing:

```python
from fastlinkage import cut, linkage

Z = linkage(embeddings, "average", metric="cosine")
coarse = cut(Z, n_clusters=20)
fine = cut(Z, distance_threshold=0.25)
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

### Reference

| | |
|---|---|
| `linkage(X, method="ward", metric=None, *, device=None)` | SciPy-format linkage matrix. `method` and `metric` as in SciPy: `"ward"`, or `"average"` with `"cosine"` or `"sqeuclidean"`. `device` is `"cuda"`, `"mps"` or `"cpu"`; by default the GPU if there is one. |
| `cut(Z, n_clusters=None, distance_threshold=None)` | Labels from a tree. Give exactly one of the two. |
| `AgglomerativeClustering(n_clusters=2, *, metric="euclidean", linkage="ward", distance_threshold=None, device=None)` | scikit-learn style estimator. Supported pairs: `ward` + `euclidean`, `average` + `cosine`, `average` + `sqeuclidean`. |

## Benchmarks

Measured on an Apple laptop GPU (PyTorch MPS, 18 CPU cores for the baselines).
Every timing builds the full tree.

| Data | Points | Linkage | fastlinkage | Baseline |
|---|---|---|---|---|
| Text embeddings, 384 features | 20,000 | Average, cosine | 0.8 s | SciPy 47 s |
| Text embeddings, 384 features | 20,000 | Ward | 1.0 s | SciPy 16 s |
| Text embeddings, 384 features | 120,000 | Average, cosine | 6.9 s | SciPy needs 58 GB |
| Text embeddings, 384 features | 120,000 | Ward | 8.7 s | fastcluster 59 min |
| Synthetic, 32 features | 100,000 | Ward | 2.2 s | fastcluster 135 s |
| Synthetic, 32 features | 200,000 | all three | 5.9 to 6.4 s | not run |
| Synthetic, 32 features | 500,000 | Average, cosine | 33 s | not run |
| Synthetic, 32 features | 500,000 | Ward | 36 s | not run |

- The text embeddings are AG News titles and descriptions, embedded with
  all-MiniLM-L6-v2.
- fastcluster's `linkage_vector` is the standard exact Ward that avoids the
  matrix. It is single-threaded.
- The same code on the CPU is 2x to 4x slower than on this GPU, so most of
  the gap to SciPy comes from the method, not the hardware.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/michael-L-i/fastlinkage/main/assets/scaling-dark.svg">
  <img src="https://raw.githubusercontent.com/michael-L-i/fastlinkage/main/assets/scaling-light.svg" alt="Exact Ward tree on 32-feature data. fastlinkage: 0.4 s at 20,000 points, 2.2 s at 100,000, 6.4 s at 200,000, 36 s at 500,000. fastcluster: 5.2 s at 20,000, 32 s at 50,000, 2.3 minutes at 100,000." width="800">
</picture>

Time grows with the square of the number of points. About a million points is
the practical limit.

### NVIDIA GPU

A free Google Colab machine: Tesla T4 with 2 virtual CPUs. Synthetic data.

| Points | Features | Ward | Average, cosine | Average, sq. Euclidean | Baseline on the same machine |
|---|---|---|---|---|---|
| 20,000 | 32 | 1.0 s | 0.8 s | 0.9 s | SciPy 23 to 24 s |
| 100,000 | 32 | 3.9 s | 4.0 s | 5.3 s | fastcluster Ward 490 s |
| 100,000 | 384 | 11.4 s | 12.7 s | 14.8 s | not run |
| 200,000 | 32 | 10.8 s | 10.0 s | 11.1 s | not run |

The trees match SciPy at 20,000 points and fastcluster at 100,000 to 1e-12.
[`notebooks/cuda_check.ipynb`](notebooks/cuda_check.ipynb) reproduces this.

### Against cuhc

[cuhc](https://github.com/shangar21/cuhc) is the closest existing package:
exact Ward on the GPU by the same rounds, with a hand-written CUDA kernel in
float32. Both on the same Tesla T4, best of three runs:

| Task | Points | Features | fastlinkage | cuhc |
|---|---|---|---|---|
| Ward | 20,000 | 32 | 0.87 s | **0.06 s** |
| Ward | 100,000 | 32 | 3.2 s | **0.94 s** |
| Ward | 200,000 | 32 | 8.3 s | **3.9 s** |
| Ward | 100,000 | 384 | **9.6 s** | 11.4 s |
| Average, cosine | 20,000 | 384 | 1.34 s | 1.27 s |
| Average, cosine | 40,000 | 384 | **2.8 s** | 4.7 s |
| Average, cosine | 100,000 | 384 | **9.6 s** | out of memory (needs 40 GB) |

| Largest relative error in merge heights, against SciPy at 20,000 points | fastlinkage | cuhc |
|---|---|---|
| Ward, 32 features | 1e-15 | 3e-7 |
| Average, cosine, 384 features | 2e-14 | 2e-4 |

cuhc is 2 to 14 times faster for Ward with few features, and the gap narrows
as the number of points grows. With 384 features the two are level. For
average linkage cuhc builds the distance matrix, so it stops between 40,000
and 100,000 points on this GPU. fastlinkage checks every merge in float64;
cuhc works in float32. Run it with
[`experiments/cuhc_comparison.py`](experiments/cuhc_comparison.py); the
numbers are in [`results/cuhc_comparison.json`](results/cuhc_comparison.json).

## Exactness

The tree is checked against independent implementations:

| Check | Result |
|---|---|
| SciPy, 5,000 and 20,000 points, all three linkages, synthetic and real embeddings | merge heights equal to 1e-12; flat clusterings identical |
| fastcluster Ward, 100,000 synthetic points | heights equal to 5e-13 |
| fastcluster Ward, 120,000 text embeddings | heights equal to 1.4e-14 |
| Duplicated rows, near-duplicates (1e-7 noise), data offset by 1e4 | equal to SciPy to 1e-13 |

This takes some care. The GPU search runs in float32, and a plain float32
search picks a slightly wrong neighbour often enough to change the tree: an
earlier version of this code was off by up to 1.6 in merge height (tallest
merge 82) on the 120,000 embeddings. The current version accepts a float32
result only when it wins by more than the possible rounding error, and
otherwise compares the close contenders in float64.

With exact ties between different pairs (an integer grid, for example) the
tree is not unique, and this package and SciPy can return different valid
trees.

## How it works

Three known ideas and two additions.

1. **Cluster summaries instead of a matrix.** For these linkages the distance
   between two clusters follows from each cluster's size, mean vector and one
   scalar:

   | Linkage | Distance between clusters A and B |
   |---|---|
   | Ward | sqrt(2 \|A\|\|B\| / (\|A\|+\|B\|)) x \|\|mean(A) - mean(B)\|\| |
   | Average, cosine | 1 - mean(A) . mean(B), rows scaled to unit length |
   | Average, squared Euclidean | \|\|mean(A) - mean(B)\|\|^2 + var(A) + var(B) |

2. **Merge in rounds.** Every pair of clusters that are each other's nearest
   neighbour merges in the same round. For these linkages that gives exactly
   the tree of the one-merge-at-a-time algorithm. A full run takes 50 to 180
   rounds.

3. **Nearest-neighbour search as one matrix product** on the GPU.

4. **Neighbour lists that survive merges.** A search returns 8 candidates and
   a radius beyond which nothing else can be. When a cluster's neighbour is
   merged away, the next one usually comes from that list with no new search,
   and a merged cluster inherits its parents' lists. This halves the search
   work (from about 3 n^2 to 1.5 n^2 pair evaluations).

5. **A float32 rounding check** with a float64 fallback, described above.

[`docs/DESIGN.md`](docs/DESIGN.md) has the details: the neighbour-list rule,
the rounding check, more measurements, and what was tried and dropped.

## Related work

None of the underlying algorithm is new, and this package does not claim it.

| Idea | Where it comes from |
|---|---|
| Merging reciprocal nearest neighbours in rounds | Murtagh (1980s); [ParChain](https://arxiv.org/abs/2106.04727) (2021, CPU); [RAC](https://arxiv.org/abs/2105.11653) (2021) |
| Average linkage from means and variances in linear memory | ParChain (CPU, kd-trees, low dimensions); [arXiv 1404.3439](https://arxiv.org/abs/1404.3439) |
| Exact Ward on the GPU | [cuhc](https://pypi.org/project/cuhc/) (2026, CUDA) |

What I could not find elsewhere is a GPU implementation of exact average
linkage (cosine or squared Euclidean) that avoids the distance matrix and
works on high-dimensional embeddings. cuhc keeps the matrix for average
linkage; ParChain's version is CPU-only and built for low dimensions. The
search was not exhaustive, so treat "first" as unconfirmed. I have also not
checked whether the neighbour-list rule (item 4) is published.

## Limits

- Three linkages only: Ward, average with cosine, average with squared
  Euclidean. Complete linkage and average linkage with plain Euclidean
  distance have no per-cluster summary and are not supported.
- Time is quadratic in the number of points.
- A float32 copy of every cluster mean lives on the GPU: 8 n d bytes (3 GB
  for a million points with 384 features). On CUDA a float64 copy is kept
  there too when it fits in a quarter of the free memory.
- When distances are below float32 resolution (cosine distance between
  almost parallel vectors), the float64 fallback does most of the work and is
  slow. A warning is raised.
- When several pairs are at exactly the same distance (integer or binary
  data, grids), more than one tree is correct. The tree returned is one of
  them, but it can differ from the one SciPy picks.
- Data that merges like a chain (evenly spaced points on a line, for
  example) needs about as many rounds as there are points, and is slow:
  3,000 such points take 8 s on an Apple GPU, against 0.5 s on the CPU.
- For Ward on an NVIDIA GPU with few features, cuhc is several times faster
  (see the comparison above).
- Tested on one NVIDIA GPU (Tesla T4) so far.

## Reproduce

```
git clone https://github.com/michael-L-i/fastlinkage && cd fastlinkage
pip install -e ".[test,bench]"
pytest                                        # exactness tests, CPU only
python experiments/correctness.py             # against SciPy
python experiments/make_embeddings.py         # AG News embeddings (176 MB)
python experiments/neighbour_lists.py         # 20k to 120k points
python experiments/large_runs.py              # fastcluster check, 200k, 500k
python experiments/make_charts.py             # the charts above
```

[`experiments/`](experiments) lists every script, and the raw numbers are in
[`results/`](results).

## Repository layout

```
src/fastlinkage/   the package
tests/             exactness tests against SciPy and scikit-learn
notebooks/         CUDA check for Google Colab
examples/          quickstart
experiments/       benchmark and correctness scripts
results/           their output (JSON)
docs/DESIGN.md     how it works, measurements, prior work
assets/            README charts
```

## License

MIT
