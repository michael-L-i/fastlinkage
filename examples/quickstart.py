"""Cluster 50,000 points three ways and time each one.

usage: python examples/quickstart.py
"""
import time

import numpy as np

from fastlinkage import AgglomerativeClustering, cut, linkage

rng = np.random.default_rng(0)
centres = rng.normal(size=(40, 64)) * 3
X = centres[rng.integers(0, 40, 50_000)] + rng.normal(size=(50_000, 64))

linkage(X[:2000], "ward")                       # warm up the GPU

t = time.perf_counter()
Z = linkage(X, "average", metric="cosine")      # scipy-format linkage matrix
print(f"full tree for {len(X):,} points: {time.perf_counter() - t:.1f} s")

labels = cut(Z, n_clusters=40)
print("clusters at k=40:", labels.max() + 1, "| largest:", np.bincount(labels).max())

labels = cut(Z, distance_threshold=0.5)
print("clusters below cosine distance 0.5:", labels.max() + 1)

t = time.perf_counter()
model = AgglomerativeClustering(n_clusters=40, linkage="ward").fit(X)
print(f"scikit-learn style Ward, 40 clusters: {time.perf_counter() - t:.1f} s")
