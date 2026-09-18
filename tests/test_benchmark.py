"""Deterministic comparison against plain kNN on scikit-learn's toy datasets.

The point is not to prove dominance but to make sure that the subspace ensemble
does not give up accuracy for its interpretability, and that complementary
selection does not fall behind the ikNN-style ranking it replaced. The full
benchmark, on fourteen datasets, is benchmarks/run_benchmark.py and
docs/benchmark.md.
"""

import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from subspaceknn import SubspaceKNNClassifier

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}
TOLERANCE_VS_KNN = 0.03
TOLERANCE_VS_RANKED = 0.01


def macro_f1(estimator, X, y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipeline = make_pipeline(StandardScaler(), estimator)
    return cross_val_score(pipeline, X, y, cv=folds, scoring="f1_macro").mean()


@pytest.mark.parametrize("name", list(DATASETS))
def test_subspace_ensembles_keep_up_with_plain_knn(name):
    X, y = DATASETS[name](return_X_y=True)
    knn = macro_f1(KNeighborsClassifier(), X, y)
    ranked = macro_f1(SubspaceKNNClassifier(selection="ranked", cv=5, max_candidates=100), X, y)
    pairs = macro_f1(SubspaceKNNClassifier(), X, y)
    mixed = macro_f1(SubspaceKNNClassifier(subspace_size=(1, 2, 3), n_subspaces=8), X, y)
    print(f"{name:>14s}: knn={knn:.3f} ranked={ranked:.3f} pairs={pairs:.3f} mixed={mixed:.3f}")
    assert pairs >= knn - TOLERANCE_VS_KNN
    assert mixed >= knn - TOLERANCE_VS_KNN
    assert pairs >= ranked - TOLERANCE_VS_RANKED
