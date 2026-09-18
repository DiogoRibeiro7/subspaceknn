"""Deterministic comparison against plain kNN on scikit-learn's toy datasets.

The point is not to prove dominance but to make sure the subspace ensemble does
not give up accuracy for its interpretability, and to record the numbers that
``docs/benchmark.md`` cites.
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
TOLERANCE = 0.05


def macro_f1(estimator, X, y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(estimator, X, y, cv=folds, scoring="f1_macro").mean()


@pytest.mark.parametrize("name", list(DATASETS))
def test_subspace_ensembles_keep_up_with_plain_knn(name):
    X, y = DATASETS[name](return_X_y=True)
    scaler = StandardScaler()
    knn = macro_f1(make_pipeline(scaler, KNeighborsClassifier()), X, y)
    pairs = macro_f1(make_pipeline(scaler, SubspaceKNNClassifier(subspace_size=2)), X, y)
    mixed = macro_f1(
        make_pipeline(scaler, SubspaceKNNClassifier(subspace_size=(1, 2, 3), n_subspaces=8)),
        X,
        y,
    )
    print(f"{name:>14s}: knn={knn:.3f} pairs={pairs:.3f} mixed(1,2,3)={mixed:.3f}")
    assert pairs >= knn - TOLERANCE
    assert mixed >= knn - TOLERANCE
