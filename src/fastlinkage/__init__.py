"""Exact hierarchical clustering on the GPU, without the n x n distance matrix."""
from .cluster import AgglomerativeClustering, cut
from .core import METHODS, linkage

__version__ = "0.1.1"
__all__ = ["linkage", "cut", "AgglomerativeClustering", "METHODS"]
