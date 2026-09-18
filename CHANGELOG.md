# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SubspaceKNNClassifier`: a scikit-learn compatible classifier that fits k-nearest-neighbour models on every small feature subspace, ranks them by cross-validated score, and combines the best by weighted soft or hard voting. Subspaces can have any size or a mix of sizes; wide data is handled by screening features to keep the candidate count within `max_candidates`.
- `explain`, returning an `Explanation` per sample with one `SubspaceVote` per subspace, and `feature_scores_` as a coarse relevance measure.
- `subspaceknn.plotting.plot_subspaces`, drawing one-, two- and three-dimensional subspaces with decision regions where possible (optional `plot` extra).
- Test suite covering scikit-learn's estimator contract for two configurations, behaviour, explanations, plotting, and an accuracy comparison against plain kNN on the iris, wine and breast-cancer datasets.
- Project infrastructure: uv-based workflow, ruff and mypy in strict mode, CI across Python 3.10 to 3.13 and three operating systems, tag-driven PyPI release with trusted publishing, Dependabot, issue and pull request templates, contributing guide, security policy and code of conduct.

[Unreleased]: https://github.com/DiogoRibeiro7/subspaceknn/commits/main
