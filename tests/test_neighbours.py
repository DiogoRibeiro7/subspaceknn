"""The tie rule of TieSharingKNeighborsClassifier, checked against an exact reference."""

from fractions import Fraction
from math import sqrt

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from subspaceknn._neighbours import TieSharingKNeighborsClassifier

METRICS = ["euclidean", "manhattan", "chebyshev"]


def _reduced(a, b, metric):
    differences = [abs(int(x) - int(y)) for x, y in zip(a, b, strict=True)]
    if metric == "euclidean":
        return sum(d * d for d in differences)
    if metric == "manhattan":
        return sum(differences)
    return max(differences)


def reference_proba(X, y, queries, k, *, metric="euclidean", weights="uniform", leave_out=None):
    """The tie rule computed one query at a time, exactly, on integer data."""
    classes = sorted(set(y.tolist()))
    rows = []
    for row, query in enumerate(queries):
        candidates = [j for j in range(len(X)) if leave_out is None or j != leave_out[row]]
        reduced = {j: _reduced(X[j], query, metric) for j in candidates}
        boundary = sorted(reduced.values())[k - 1]
        closer = [j for j in candidates if reduced[j] < boundary]
        tied = [j for j in candidates if reduced[j] == boundary]
        share = Fraction(k - len(closer), len(tied))
        weight = {j: Fraction(1) for j in closer} | dict.fromkeys(tied, share)
        if weights == "distance":
            at_zero = [j for j in candidates if reduced[j] == 0]
            if at_zero:
                weight = dict.fromkeys(at_zero, Fraction(1))
            else:
                root = sqrt if metric == "euclidean" else float
                weight = {j: float(w) / root(reduced[j]) for j, w in weight.items()}
        totals = [sum(w for j, w in weight.items() if y[j] == c) for c in classes]
        rows.append(
            [
                float(Fraction(t) / sum(totals)) if weights == "uniform" else t / sum(totals)
                for t in totals
            ]
        )
    return np.array(rows, dtype=np.float64)


def tied_data(seed, n=60, n_features=2, levels=4, n_classes=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, levels, size=(n, n_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n)
    return X, y


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_leave_one_out_follows_the_tie_rule(metric, weights, k):
    X, y = tied_data(seed=k)
    model = TieSharingKNeighborsClassifier(k, weights=weights, metric=metric).fit(X, y)
    expected = reference_proba(
        X, y, X, k, metric=metric, weights=weights, leave_out=np.arange(len(X))
    )
    assert_allclose(model.leave_one_out_proba(), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_predictions_follow_the_tie_rule(metric, weights):
    X, y = tied_data(seed=7)
    queries = np.random.default_rng(8).integers(-1, 5, size=(30, 2)).astype(np.float64)
    model = TieSharingKNeighborsClassifier(5, weights=weights, metric=metric).fit(X, y)
    expected = reference_proba(X, y, queries, 5, metric=metric, weights=weights)
    assert_allclose(model.predict_proba(queries), expected, rtol=1e-12, atol=1e-15)


def test_large_tie_groups_are_found_completely():
    # Five distinct values, each repeated 40 times: every tie group is far larger
    # than the first candidate list, so the search has to widen several times.
    X = np.repeat(np.arange(5.0), 40).reshape(-1, 1)
    y = np.random.default_rng(0).integers(0, 2, size=len(X))
    model = TieSharingKNeighborsClassifier(5).fit(X, y)
    expected = reference_proba(X, y, X, 5, leave_out=np.arange(len(X)))
    assert_allclose(model.leave_one_out_proba(), expected, rtol=1e-12)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_votes_do_not_depend_on_row_order(weights):
    X, y = tied_data(seed=3, n=80)
    permutation = np.random.default_rng(4).permutation(len(X))
    first = TieSharingKNeighborsClassifier(5, weights=weights).fit(X, y)
    second = TieSharingKNeighborsClassifier(5, weights=weights).fit(X[permutation], y[permutation])
    assert_array_equal(second.leave_one_out_proba(), first.leave_one_out_proba()[permutation])
    queries = np.random.default_rng(5).integers(0, 4, size=(25, 2)).astype(np.float64)
    assert_array_equal(second.predict_proba(queries), first.predict_proba(queries))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_without_ties_it_matches_scikit_learn(weights):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.integers(0, 3, size=30)
    fast = TieSharingKNeighborsClassifier(4, weights=weights).fit(X, y).leave_one_out_proba()
    for row in range(30):
        model = KNeighborsClassifier(n_neighbors=4, weights=weights)
        model.fit(np.delete(X, row, axis=0), np.delete(y, row))
        assert_allclose(fast[row], model.predict_proba(X[row : row + 1])[0])
    queries = rng.normal(size=(10, 2))
    ours = TieSharingKNeighborsClassifier(4, weights=weights).fit(X, y).predict_proba(queries)
    theirs = KNeighborsClassifier(4, weights=weights).fit(X, y).predict_proba(queries)
    assert_allclose(ours, theirs)


def test_points_at_distance_zero_take_all_the_weight():
    X = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 0, 0])
    model = TieSharingKNeighborsClassifier(3, weights="distance").fit(X, y)
    assert_allclose(model.leave_one_out_proba()[0], [0.0, 1.0])
    assert_allclose(model.predict_proba([[0.0]])[0], [0.0, 1.0])


def test_labels_and_predictions_keep_their_type():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [1.2]])
    y = np.array(["a", "a", "b", "b", "b"])
    model = TieSharingKNeighborsClassifier(3).fit(X, y)
    assert_array_equal(model.classes_, ["a", "b"])
    assert_array_equal(model.predict([[0.05], [1.05]]), ["a", "b"])


def test_other_metrics_use_the_search_distances():
    X, y = tied_data(seed=9)
    X = X + 1.0  # cosine distance is undefined at the origin
    proba = TieSharingKNeighborsClassifier(5, metric="cosine").fit(X, y).leave_one_out_proba()
    assert proba.shape == (len(X), 3)
    assert_allclose(proba.sum(axis=1), 1.0)


def test_too_few_training_points_raise():
    model = TieSharingKNeighborsClassifier(5).fit(np.zeros((5, 1)), np.zeros(5))
    with pytest.raises(ValueError, match="Expected n_neighbors <= 4"):
        model.leave_one_out_proba()
