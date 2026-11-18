# Experiments

Scripts behind the numbers in the main README. Each one writes a JSON file of
the same name to [`../results`](../results). Run them from the repository root
after `pip install -e ".[test,bench]"`.

| Script | What it measures | Needs |
|---|---|---|
| `correctness.py` | Same tree as SciPy, three linkages, 5,000 and 20,000 synthetic points | |
| `scaling.py` | Time against SciPy and fastcluster as the number of points grows | |
| `make_embeddings.py` | Builds `data/agnews_minilm.npy`: 120,000 AG News items embedded with all-MiniLM-L6-v2 (176 MB, not in the repository) | |
| `embeddings.py` | Same tree as SciPy on 20,000 embeddings; timing on all 120,000 | embeddings |
| `neighbour_lists.py` | The package against the brute-force baseline: time, search work, list hit rates | embeddings |
| `large_runs.py` | fastcluster Ward check at 100,000 points; timings at 200,000 and 500,000 | |
| `ward_embeddings_check.py` | fastcluster Ward on all 120,000 embeddings (about an hour) | embeddings |
| `pruning_bound.py` | A search-pruning idea that was measured and dropped | embeddings |
| `cuhc_comparison.py` | The package against cuhc on the same NVIDIA GPU: time, height error, where cuhc's matrix stops fitting | CUDA, `pip install cuhc` |
| `make_charts.py` | Draws the charts in `assets/` from the result files | |

`brute_force_baseline.py` is the first version of the algorithm: a full search
for every cluster that lost its neighbour, every round, with no neighbour
lists and no float32 rounding check. It is kept only as the comparison point
for `neighbour_lists.py` and `ward_embeddings_check.py`, and it is not exact
on large inputs.

The timings were taken on an Apple laptop GPU. The largest run (500,000
points) takes about a minute and a few GB of memory.
