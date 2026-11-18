# Contributing

Bug reports and pull requests are welcome. The most useful reports are a
tree that differs from SciPy's, or a timing on hardware not listed in the
README.

## Set up

```
git clone https://github.com/michael-L-i/fastlinkage
cd fastlinkage
pip install -e ".[dev]"
```

## Before a pull request

```
ruff check .
pytest
```

The tests run on the CPU and on every GPU PyTorch can see. CI runs them on
the CPU on Linux, macOS and Windows.

The rule for any change to the algorithm: the tree must stay the same as
SciPy's. `tests/test_linkage.py` checks that on small inputs, including
duplicates and near-ties; `experiments/correctness.py` and
`experiments/large_runs.py` check it at 20,000 and 100,000 points.

[`docs/DESIGN.md`](docs/DESIGN.md) explains how the code works and lists the
optimisations that were tried and dropped.

## Releasing

1. Set the version in `src/fastlinkage/__init__.py` and `CITATION.cff`, and
   add a section to `CHANGELOG.md`.
2. Publish a GitHub release with the tag `vX.Y.Z`. The `publish` workflow
   builds the package and uploads it to PyPI.
