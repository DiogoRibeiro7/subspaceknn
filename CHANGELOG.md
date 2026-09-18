# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation site built with MkDocs and Material for MkDocs, published to GitHub Pages: a user guide, the method and benchmark notes, an API reference generated from the docstrings, and the changelog.
- Complementary selection, the new default (`selection="complementary"`). The ensemble is built by greedy forward selection with replacement on the candidates' out-of-fold probabilities: each step adds the subspace that most reduces the ensemble's class-balanced Brier score, `n_subspaces` caps the number of distinct subspaces, and the weights are vote counts. Across fourteen benchmark datasets it improves mean macro-F1 over ranking subspaces individually by 1.8 to 2.6 points, depending on the subspace sizes and the number of subspaces.
- Exact leave-one-out out-of-fold probabilities from a single neighbour query per subspace, the new default `cv="loo"`. It is faster than k-fold scoring and needs no random split.
- Parameters `selection`, `max_votes` and `balance_classes`, and the fitted attribute `selection_path_`, which records the out-of-fold loss after each vote.
- `benchmarks/run_benchmark.py`, which reproduces the fourteen-dataset benchmark in `docs/benchmark.md`, including an ablation of the two ingredients.
- References section in the README and method note, starting with Brett Kennedy's ikNN article, which the method builds on.

### Changed

- The ikNN-style behaviour of 0.1.0 is now `selection="ranked"`; `SubspaceKNNClassifier(selection="ranked", cv=5, max_candidates=100)` reproduces the 0.1.0 configuration up to how subspace scores are averaged (next item).
- Subspace scores are computed on pooled out-of-fold predictions rather than averaged over folds, so any scorer works, including probability-based ones. A splitter passed as `cv` must partition the samples.
- `max_candidates` defaults to 1000 instead of 100, which the cheaper scoring makes affordable.
- `weighting` applies to ranked selection only; complementary selection weights by vote counts.
- Plot panel titles show each subspace's weight next to its score.

## [0.1.0] - 2026-09-18

### Added

- `SubspaceKNNClassifier`: a scikit-learn compatible classifier that fits k-nearest-neighbour models on every small feature subspace, ranks them by cross-validated score, and combines the best by weighted soft or hard voting. Subspaces can have any size or a mix of sizes; wide data is handled by screening features to keep the candidate count within `max_candidates`.
- `explain`, returning an `Explanation` per sample with one `SubspaceVote` per subspace, and `feature_scores_` as a coarse relevance measure.
- `subspaceknn.plotting.plot_subspaces`, drawing one-, two- and three-dimensional subspaces with decision regions where possible (optional `plot` extra).
- Test suite covering scikit-learn's estimator contract for two configurations, behaviour, explanations, plotting, and an accuracy comparison against plain kNN on the iris, wine and breast-cancer datasets.
- Project infrastructure: uv-based workflow, ruff and mypy in strict mode, CI across Python 3.10 to 3.13 and three operating systems, tag-driven PyPI release with trusted publishing, Dependabot, issue and pull request templates, contributing guide, security policy and code of conduct.

[Unreleased]: https://github.com/DiogoRibeiro7/subspaceknn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/subspaceknn/releases/tag/v0.1.0
