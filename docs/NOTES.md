# Development notes

The working log of this project, kept as written (the project was called ward-gpu
at the time). The README is the summary; this has the detail.

Exact hierarchical clustering without the n x n distance matrix, with the
nearest-neighbour search on the GPU. A prototype, tested on an Apple GPU
through PyTorch.

```python
from fastlinkage.core import linkage

Z = linkage(X, method="ward")                 # scipy-format linkage matrix
Z = linkage(X, method="average_cosine")       # scipy: linkage(X, "average", metric="cosine")
Z = linkage(X, method="average_sqeuclidean")  # scipy: linkage(X, "average", metric="sqeuclidean")
```

## Status (2026-11-11)

- Version 2 (`src/fastlinkage/core.py`, current): neighbour-list cache plus a
  float32 rounding check. About half the search work of version 1, 1.3x to
  1.7x faster from 100,000 points up, and exact in cases where version 1 was
  slightly off. See "Version 2" below.
- Version 1 (`experiments/baseline_v1.py`, kept as the baseline): brute-force
  search on the GPU every round. The tables under "Results" are from it.
- What it is: a fast, exact tool for a known algorithm. Not a new algorithm.
  Ward on the GPU already exists (cuhc); average linkage on the GPU without
  the matrix was not found anywhere else.
- Who it is for: exact hierarchical clustering of roughly 30,000 to 1,000,000
  embeddings, cut at a distance threshold. Planned shape: a pip package with
  scipy's `linkage` and a scikit-learn style estimator.
- Open before a release: CUDA test, comparison with cuhc, a better name (Ward
  is the part that is not new), the estimator class.

## Version 2

Two changes to version 1.

**Neighbour lists that survive merges.** A full search returns 16 candidates
per cluster and a radius: every cluster that is not a candidate is at least
that far away. Because the linkages are reducible, a later cluster that
contains none of the candidates is still at least that far away. So when a
cluster loses its nearest neighbour, the nearest of its remaining list
members is the true nearest as long as it is inside the radius, and no search
is needed. A merged cluster inherits both parents' lists; its radius is the
size-weighted mean of the parents' radii for average linkage, and a
Lance-Williams bound for Ward. A cluster goes back to the full search only
when its list runs dry. I have not checked whether this rule is published.

**A rounding check.** The GPU search is float32. A candidate is accepted only
if it beats every other column by more than a bound on the float32 error.
Otherwise every column inside the bound is fetched and compared in float64.
This closes the tie gap of version 1. Identical rows are merged up front.

Version 2 against version 1, Apple GPU (exp6, exp7):

| Data | Points | Method | Version 1 | Version 2 | Pair evaluations / n^2 |
|---|---|---|---|---|---|
| Synthetic, 32 features | 100,000 | Ward | 2.8 s | 2.2 s | 3.2 to 1.5 |
| Synthetic, 32 features | 100,000 | Average, cosine | 2.8 s | 2.2 s | 3.2 to 1.4 |
| Synthetic, 32 features | 100,000 | Average, sq. Euclidean | 3.1 s | 2.3 s | 3.2 to 1.4 |
| Embeddings, 384 features | 120,000 | Ward | 10.5 s | 8.7 s | 2.7 to 1.7 |
| Embeddings, 384 features | 120,000 | Average, cosine | 9.6 s | 6.9 s | 2.6 to 1.4 |
| Embeddings, 384 features | 120,000 | Average, sq. Euclidean | 10.2 s | 7.3 s | 2.6 to 1.4 |
| Synthetic, 32 features | 200,000 | all three | 8.2 to 8.8 s | 5.9 to 6.4 s | 3.0 to 1.5 |
| Synthetic, 32 features | 500,000 | Ward | 57 s | 36 s | 3.0 to 1.7 |
| Synthetic, 32 features | 500,000 | Average, cosine | 65 s | 33 s | 3.1 to 1.5 |

- At 20,000 points the two versions take the same time: the saved searches
  are offset by the bookkeeping.
- The first round alone costs n^2 pair evaluations (every point against every
  point), so 1.0 is the floor for brute-force search. Version 2 is at 1.4 to
  1.7. The lists settle 47% to 81% of new clusters and 62% to 91% of clusters
  that lost their neighbour.
- Deeper lists (32 or 64 candidates) cut the work to about 1.2 but cost more
  CPU time than they save.

Exactness of version 2:

- Same trees as scipy on all exp1 and exp6 cases at 5,000 and 20,000 points.
- 100,000 synthetic points, Ward: heights match fastcluster's exact Ward to
  5e-13 (fastcluster 135 s, this 2.2 s).
- Version 1 was wrong in a few merges at scale. On 100,000 synthetic points
  (average, squared Euclidean) it merged a cluster with a neighbour 1e-6
  further away than the nearest, in 4 merges. On the 120,000 embeddings it
  differs from version 2 for Ward (1.6 in one height) and average cosine
  (5e-3). exp8 ran fastcluster's exact Ward on the same 120,000 embeddings
  (3,561 s): version 2 matches it to 1.4e-14, version 1 is off by up to 1.6
  (tallest merge 82), with 20,930 of the sorted heights differing.
- Hard inputs, 8,000 to 20,000 points, against scipy: rows duplicated 10
  times, near-duplicates (1e-7 noise) and data offset by 1e4 all match to
  1e-13 or better. Version 1 was off by up to 5e-8 on these and up to 20x
  slower.
- On an integer lattice (thousands of exact ties) the heights differ from
  scipy's for Ward and average squared Euclidean. With exact ties the tree is
  not unique.
- When distances are below float32 resolution (cosine on vectors that are
  almost parallel), the float64 check does most of the work and it is slow:
  3 s for 2,000 such points. A warning is raised.

## What is new here and what is not

Nothing in the algorithm is new. The pieces and where they come from:

| Piece | Prior work |
|---|---|
| Merge all reciprocal nearest neighbours in the same round; exact for reducible linkages | Murtagh (1980s); [ParChain](https://arxiv.org/abs/2106.04727) (2021, CPU); [RAC](https://arxiv.org/abs/2105.11653) (2021, distributed CPU, graph input); [cuhc](https://pypi.org/project/cuhc/) (2026, GPU, Ward only) |
| Average linkage with squared Euclidean distance from cluster means and variances, in linear memory | ParChain's "avg-2" (CPU, kd-tree search, benchmarked up to 16 dimensions); identity also in [arXiv 1404.3439](https://arxiv.org/abs/1404.3439) |
| Average linkage with cosine distance from cluster mean vectors | arXiv 1404.3439 (identity only); the group-average method in the Manning, Raghavan and Schutze IR book (2008) |
| Exact Ward on the GPU without the matrix | cuhc, released 2026-10-19: 0.53 s for 100k points and 51 s for 1M on an RTX 3080 |

What a background search (19 sources read or searched) did not find: a GPU
implementation of exact average linkage that avoids the matrix. cuhc, the
closest GPU package, stores the dense matrix for average linkage (about 50,000
points on a 12 GB GPU) and says no exact matrix-free algorithm is known.
ParChain's matrix-free average linkage relies on kd-trees, which suit low
dimensions, and does not cover cosine.

So the honest claim is: to my knowledge, the first GPU implementation of exact
matrix-free average linkage for cosine and squared-Euclidean distance, which
also works on high-dimensional embeddings. The search was not exhaustive
(bioinformatics UPGMA tools were not covered), so "first" is unconfirmed.

Not covered: complete linkage, and average linkage with plain Euclidean
distance. Neither has a per-cluster summary.

## How it works

Cluster distances from summaries:

| Method | Distance between clusters A and B |
|---|---|
| Ward | sqrt(2 \|A\|\|B\| / (\|A\|+\|B\|)) x \|\|mean(A) - mean(B)\|\| |
| Average, cosine | 1 - mean(A) . mean(B), rows scaled to unit length |
| Average, squared Euclidean | \|\|mean(A) - mean(B)\|\|^2 + var(A) + var(B) |

Each round:

1. Every cluster that needs it finds its nearest cluster. This is one matrix
   product on the GPU (float32), which returns a few candidates per cluster.
2. The CPU picks the true nearest of the candidates in float64.
3. Every pair of clusters that are each other's nearest merges, all at once.
4. Only new clusters and clusters whose neighbour was merged look again
   (version 2: in their neighbour list first, then by a full search).

## Results (version 1)

Apple GPU (PyTorch MPS). All timings build the full tree.

### Same tree as scipy

- Synthetic (exp1): nine cases, three methods on 5k and 20k points, clustered
  and unstructured. Merge heights match scipy to 2e-13 relative or better;
  clusterings cut at 2, 5, 20, 100 and 1000 clusters are identical.
- Real text embeddings (exp5): first 20,000 AG News embeddings, 384 features.
  Heights match to 2e-14 absolute; the same five cuts are identical for all
  three methods.

### Real text embeddings (exp5)

AG News, 120,000 news items, all-MiniLM-L6-v2, 384 features.

| Points | Method | This repo | scipy |
|---|---|---|---|
| 20,000 | Average, cosine | 0.78 s | 47.4 s |
| 20,000 | Ward | 0.96 s | 15.9 s |
| 20,000 | Average, sq. Euclidean | 0.89 s | 15.8 s |
| 120,000 | Average, cosine | 9.8 s | needs a 58 GB matrix |
| 120,000 | Ward | 10.4 s | needs a 58 GB matrix |
| 120,000 | Average, sq. Euclidean | 9.7 s | needs a 58 GB matrix |

The same code on the CPU (18 cores) takes 42.2 s for average cosine at 120,000
points, so the GPU is worth 4.3x at 384 features.

### Synthetic, 32 features (exp2)

| Points | Ward | Average, cosine | Average, sq. Euclidean | CPU baselines |
|---|---|---|---|---|
| 20,000 | 0.33 s | 0.46 s | 0.42 s | scipy 3.9 to 4.8 s; fastcluster Ward 5.2 s |
| 50,000 | 0.91 s | 0.88 s | 1.01 s | fastcluster Ward 31.9 s |
| 100,000 | 2.7 s | 2.5 s | 2.6 s | fastcluster Ward 129 s (first version's run) |
| 200,000 | 8.8 s | 8.2 s | 8.8 s | not run |
| 500,000 | 57 s | 65 s | not run | not run |

- fastcluster's `linkage_vector` is the standard memory-saving exact Ward on
  the CPU (single thread). scipy needs the matrix and was run at 20,000 only.
- The same code on the CPU takes 2.1 to 2.5 s at 50,000 points, so at 32
  features the GPU is worth about 2.4x.
- 48 to 93 rounds; total pair evaluations stay near 3 x n^2, so time grows
  with n^2. Past 200,000 points it grows a little faster than that because the
  distance blocks get narrow.
- 100,000 Higgs rows (28 features, real, exp3, earlier version of the code):
  2.3 to 5.5 s, 74 to 128 rounds.

## Optimisations tried

| Change | Effect | Kept |
|---|---|---|
| Fold the squared distance into the matrix product (two extra columns) and take one argmin per column segment instead of a top-k | 100k Ward 25.6 s to 5.6 s; average 16.5 s to 2.6 s | Yes |
| Ward: sort columns by cluster size so the size factor is constant inside a segment, which removes a full pass per block | 100k Ward 5.6 s to 2.7 s | Yes |
| Float64 check with threaded torch instead of numpy | check time roughly halved (1.6 s to 0.8 s at 120k x 384) | Yes |
| Lazy re-search: a cluster that lost its neighbour searches again only when another cluster points at it | 25% fewer pair evaluations, but 1.1x to 2.4x slower: many small GPU batches | Option, off |
| Pruning with a low-dimensional bound (leading principal coordinates plus tail norm), exact | On 120k x 384: 21 to 25 s against 9.8 s for brute force. The bound rules out 99.9% of columns (exp4), but the masking and gathering cost more than the product they save | Removed |

Where it stands: the search runs at about 13 billion pair evaluations per
second at 32 features, against 16 billion for the bare matrix product on this
GPU, and at about 3.4 TFLOPS at 384 features. Brute force is near what this
GPU can do. Further gains would need a faster GPU or fewer than 3 x n^2 pair
evaluations.

## Limits

- Time is quadratic in the number of points.
- Unstructured data can need more rounds (173 on 5,000 standard-normal points
  with average, squared Euclidean).
- With exact ties between different pairs the tree is not unique, and this
  code and scipy can pick different ones.
- Version 2 keeps a float32 copy of every cluster mean on the GPU: 8 n d
  bytes (3 GB for a million points with 384 features).
- Not tested on CUDA, and not compared against cuhc on the same hardware.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy torch fastcluster scikit-learn \
    sentence-transformers datasets
.venv/bin/python experiments/exp1_correctness.py
.venv/bin/python experiments/exp2_scaling.py
.venv/bin/python experiments/make_embeddings.py && .venv/bin/python experiments/exp5_embeddings.py
.venv/bin/python experiments/exp6_cache.py
.venv/bin/python experiments/exp7_large.py
```
