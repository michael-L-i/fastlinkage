# Design notes

How fastlinkage works in more detail than the README, what was measured along
the way, and what was tried and dropped. All timings are from an Apple laptop
GPU (PyTorch MPS) and build the full tree.

- [The algorithm](#the-algorithm)
- [Neighbour lists](#neighbour-lists)
- [Float32 search, float64 decisions](#float32-search-float64-decisions)
- [The GPU search](#the-gpu-search)
- [Measurements](#measurements)
- [Optimisations tried](#optimisations-tried)
- [Prior work](#prior-work)
- [Code map](#code-map)

## The algorithm

For three linkages the distance between two clusters follows from a small
summary of each cluster:

| Linkage | Distance between clusters A and B | Summary |
|---|---|---|
| Ward | sqrt(2 \|A\|\|B\| / (\|A\|+\|B\|)) x \|\|mean(A) - mean(B)\|\| | size, mean |
| Average, cosine | 1 - mean(A) . mean(B), rows scaled to unit length | mean |
| Average, squared Euclidean | \|\|mean(A) - mean(B)\|\|^2 + var(A) + var(B) | mean, variance |

Here var is the mean squared distance from a cluster's points to its mean.
No distance matrix is needed: a merge only updates the summaries.

All three linkages are reducible: merging two clusters never brings the
result closer to a third cluster than both parts were. So every pair of
clusters that are each other's nearest neighbour can be merged in the same
round, and the tree is the one the usual one-merge-at-a-time algorithm
builds. Each round:

1. Every cluster without a known nearest neighbour gets one, from its
   neighbour list if possible and otherwise from a full search on the GPU.
2. Every pair of clusters that are each other's nearest merges.
3. Clusters whose nearest neighbour was just merged away lose it. Everyone
   else keeps theirs, which is still correct by reducibility.

A run takes 50 to 180 rounds. Unstructured data needs more of them (173 on
5,000 standard-normal points with average, squared Euclidean).

## Neighbour lists

A full search returns 16 candidates per cluster and a radius: every cluster
that is not a candidate is at least that far away. By reducibility, a later
cluster that contains none of the candidates is still at least that far away.
So when a cluster loses its nearest neighbour, the nearest of its remaining
list members is the true nearest as long as it is inside the radius, and no
search is needed.

A merged cluster inherits both parents' lists. Its radius is the
size-weighted mean of the parents' radii for average linkage, and a
Lance-Williams bound for Ward. A cluster goes back to the full search only
when its list runs dry.

Effect, measured against a baseline that does a full search every time
(`experiments/neighbour_lists.py`, `experiments/large_runs.py`):

| Data | Points | Linkage | Full search every time | With neighbour lists | Pair evaluations / n^2 |
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

- The first round alone costs n^2 pair evaluations (every point against
  every point), so 1.0 is the floor. The lists bring the total to 1.4 to 1.7.
- The lists settle 47% to 81% of new clusters and 62% to 91% of clusters that
  lost their neighbour.
- At 20,000 points there is no gain: the saved searches are offset by the
  bookkeeping.
- Deeper lists (32 or 64 candidates) cut the work to about 1.2 n^2 but cost
  more CPU time than they save.

I have not checked whether this rule is published elsewhere.

## Float32 search, float64 decisions

The GPU search runs in float32. Merge decisions and heights are float64:

- The CPU recomputes the distance to every candidate in float64.
- A candidate is accepted only if it beats every non-candidate column by more
  than a bound on the float32 rounding error.
- Otherwise every column inside the bound is fetched and compared in float64.
- Identical rows are merged up front at height zero.
- On CUDA the float64 work runs on the GPU, which has float64. Apple GPUs do
  not, so there it runs on the CPU. On a machine with a weak CPU this matters:
  on a Colab T4 with 2 CPU cores, moving it to the GPU took a 100,000 x 384
  run from 31 s to 13 s (`results/cuda_t4_colab.json`).

This matters. The baseline without the check picks a slightly wrong neighbour
often enough to change the tree at scale:

| Input | Baseline without the check | With the check |
|---|---|---|
| 100,000 synthetic points, average sq. Euclidean | 4 merges with a neighbour 1e-6 further than the nearest | matches |
| 120,000 embeddings, Ward, against fastcluster (3,561 s) | off by up to 1.6 in height (tallest merge 82); 20,930 sorted heights differ | equal to 1.4e-14 |
| Rows duplicated 10 times, near-duplicates (1e-7 noise), data offset by 1e4, against SciPy | off by up to 5e-8, up to 20x slower | equal to 1e-13 |

Two cases where the result or the speed differs from what you might expect:

- With exact ties between different pairs (an integer lattice) the tree is
  not unique, and the heights can differ from SciPy's.
- When distances are below float32 resolution (cosine on almost parallel
  vectors) the float64 check does most of the work and is slow: 3 s for
  2,000 such points. A warning is raised.

## The GPU search

The float32 distance from a block of query clusters to all active clusters is
one matrix product: the squared norms (and the variance, for average
linkage) are folded into extra columns. The columns are cut into about 256
equal segments, and one pass takes the minimum of each segment. The 16
segments with the smallest minima supply the candidates. A second look at
only those segments finds their runners-up, which gives the radius.

Ward's distance also has the factor |A||B| / (|A|+|B|). While there are at
most 24 distinct cluster sizes, the columns are sorted by size and padded so
each segment holds one size, and the factor is applied to the segment minima
instead of to the whole block.

The search runs at about 13 billion pair evaluations per second at 32
features, against 16 billion for the bare matrix product on the same GPU. A
float32 copy of every cluster mean lives on the device: 8 n d bytes.

## Measurements

The README has the headline table. Additional numbers:

Same tree as SciPy (`experiments/correctness.py`, `experiments/embeddings.py`):

- Synthetic: three linkages on 5,000 and 20,000 points, clustered and
  unstructured. Heights match to 2e-13 relative or better; clusterings cut at
  2, 5, 20, 100 and 1,000 clusters are identical.
- First 20,000 AG News embeddings (384 features): heights match to 2e-14;
  the same five cuts are identical.

GPU against CPU, same code:

| Data | Points | Linkage | Apple GPU | CPU, 18 cores |
|---|---|---|---|---|
| Embeddings, 384 features | 120,000 | Average, cosine | 9.8 s | 42.2 s |
| Synthetic, 32 features | 50,000 | all three | 0.9 to 1.0 s | 2.1 to 2.5 s |

(These two rows were measured before the neighbour lists were added.)

## Optimisations tried

| Change | Effect | Kept |
|---|---|---|
| Fold the squared distance into the matrix product and take one argmin per column segment instead of a top-k | 100k Ward 25.6 s to 5.6 s; average 16.5 s to 2.6 s | Yes |
| Ward: sort columns by cluster size so the size factor is constant inside a segment | 100k Ward 5.6 s to 2.7 s | Yes |
| Float64 check with threaded torch instead of numpy | check time roughly halved (1.6 s to 0.8 s at 120k x 384) | Yes |
| Neighbour lists that survive merges | 1.3x to 1.7x faster from 100k points up | Yes |
| Lazy re-search: a cluster that lost its neighbour searches again only when another cluster points at it | 25% fewer pair evaluations, but 1.1x to 2.4x slower: many small GPU batches | No |
| Pruning with a low-dimensional bound (leading principal coordinates plus tail norm), exact | On 120k x 384: 21 to 25 s against 9.8 s. The bound rules out 99.9% of columns (`experiments/pruning_bound.py`), but the masking and gathering cost more than the product they save | No |

## Prior work

Nothing in the underlying algorithm is new.

| Piece | Prior work |
|---|---|
| Merge all reciprocal nearest neighbours in the same round; exact for reducible linkages | Murtagh (1980s); [ParChain](https://arxiv.org/abs/2106.04727) (2021, CPU); [RAC](https://arxiv.org/abs/2105.11653) (2021, distributed CPU, graph input); [cuhc](https://pypi.org/project/cuhc/) (2026, GPU, Ward only) |
| Average linkage with squared Euclidean distance from means and variances, in linear memory | ParChain's "avg-2" (CPU, kd-tree search, benchmarked up to 16 dimensions); identity also in [arXiv 1404.3439](https://arxiv.org/abs/1404.3439) |
| Average linkage with cosine distance from mean vectors | arXiv 1404.3439 (identity only); the group-average method in the Manning, Raghavan and Schutze IR book (2008) |
| Exact Ward on the GPU without the matrix | cuhc, released 2026-10-19: 0.53 s for 100k points and 51 s for 1M on an RTX 3080 |

A background search (19 sources) did not find a GPU implementation of exact
average linkage that avoids the matrix. cuhc stores the dense matrix for
average linkage (about 50,000 points on a 12 GB GPU). ParChain's matrix-free
average linkage relies on kd-trees, which suit low dimensions, and does not
cover cosine. The search was not exhaustive (bioinformatics UPGMA tools were
not covered), so "first" is unconfirmed.

## Code map

| File | Contents |
|---|---|
| `src/fastlinkage/core.py` | `linkage`: input preparation and the round loop |
| `src/fastlinkage/_clusters.py` | Cluster summaries, nearest neighbours, neighbour lists, merging |
| `src/fastlinkage/_search.py` | The float32 search on the device |
| `src/fastlinkage/_distance.py` | Float64 cluster distances from summaries |
| `src/fastlinkage/_tree.py` | Merges to SciPy's linkage matrix |
| `src/fastlinkage/cluster.py` | `cut` and `AgglomerativeClustering` |
