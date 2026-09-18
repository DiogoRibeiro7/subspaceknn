import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from subspaceknn import SubspaceKNNClassifier
from subspaceknn.plotting import plot_subspaces


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_one_panel_per_subspace(iris, size):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=size, n_subspaces=2).fit(X, y)
    fig = plot_subspaces(clf, X, y, sample=X[0])
    try:
        assert len(fig.axes) == 2
        if size == 3:
            assert all(ax.name == "3d" for ax in fig.axes)
        else:
            assert all(ax.name == "rectilinear" for ax in fig.axes)
    finally:
        plt.close(fig)


def test_n_subspaces_limits_panels_and_large_subspaces_fall_back(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=4, n_subspaces=None).fit(X, y)
    fig = plot_subspaces(clf, X, y, n_subspaces=1)
    try:
        assert len(fig.axes) == 1
        assert "first two of 4 features" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_unfitted_estimator_raises(iris):
    X, y = iris
    with pytest.raises(NotFittedError):
        plot_subspaces(SubspaceKNNClassifier(), X, y)
