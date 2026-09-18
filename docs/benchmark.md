# Accuracy against plain kNN

The question is whether the interpretable ensemble gives up accuracy. `tests/test_benchmark.py` answers it on scikit-learn's toy datasets and asserts that the answer stays within a fixed tolerance; this note records the numbers.

## Protocol

- Datasets: `load_iris` (150 samples, 4 features, 3 classes), `load_wine` (178, 13, 3), `load_breast_cancer` (569, 30, 2).
- Features standardised with `StandardScaler` inside a `Pipeline`, so that every model sees the same geometry.
- Five-fold stratified cross-validation, shuffled with `random_state=0`, scored by macro-F1.
- Baseline: `KNeighborsClassifier()` with its defaults (5 neighbours, uniform weights).
- Subspace ensembles at their defaults except the subspace sizes: `SubspaceKNNClassifier(subspace_size=2)` and `SubspaceKNNClassifier(subspace_size=(1, 2, 3), n_subspaces=8)`.

## Results

| Dataset | kNN | Subspaces of size 2 | Sizes 1, 2, 3 with 8 voters |
| --- | ---: | ---: | ---: |
| iris | 0.953 | 0.953 | 0.953 |
| wine | 0.960 | 0.945 | 0.967 |
| breast cancer | 0.962 | 0.946 | 0.943 |

The test asserts that each ensemble scores at least `kNN - 0.05` on every dataset.

## Reading the numbers

- On iris the three models coincide: with four features and default settings the best pairs already carry the signal, and a kNN on all four does no better.
- On wine, mixing sizes helps: several one- and three-feature subspaces score higher than the best pairs, and the ensemble edges past the full-space kNN.
- On breast cancer the ensembles trail by less than two points. Thirty features give 435 pairs, above the default cap of 100 candidates, so screening keeps the 14 features whose one-dimensional models score best (91 pairs). Raising `max_candidates`, or letting more subspaces vote, is the natural lever if those two points matter more than the explanation.

The point of the comparison is not that subspace ensembles beat kNN. It is that on these datasets you can have a prediction made of a handful of two-dimensional pictures at essentially the accuracy of the opaque model.
