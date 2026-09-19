# API reference

The package exports the classifier and its explanation types from `subspaceknn`. Plotting lives in `subspaceknn.plotting` and needs the optional `plot` extra.

::: subspaceknn.SubspaceKNNClassifier

::: subspaceknn.Explanation

::: subspaceknn.SubspaceVote

::: subspaceknn.plotting.plot_subspaces

## Subspace models

Every entry of `estimators_` is one of these. It is not exported from the package: create a `SubspaceKNNClassifier` instead.

::: subspaceknn._neighbours.TieSharingKNeighborsClassifier
    options:
      show_root_full_path: false
