"""Property-based tests of the selection invariants on generated data full of ties.

Features are small integers, so equidistant neighbours are everywhere: the data
that made selection depend on the platform before the tie rule. The Hypothesis
profile is derandomised so that every run, locally and in CI, tries the same
examples.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from subspaceknn import SubspaceKNNClassifier

SETTINGS = settings(
    derandomize=True,
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def problems(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n_samples = draw(st.integers(12, 40))
    n_features = draw(st.integers(2, 4))
    n_classes = draw(st.integers(2, 3))
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n_samples, n_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n_samples)
    params = {
        "n_neighbors": draw(st.sampled_from([1, 3, 5])),
        "subspace_size": draw(st.sampled_from([1, 2, (1, 2)])),
        "n_subspaces": draw(st.sampled_from([1, 2, 3, None])),
        "voting": draw(st.sampled_from(["soft", "hard"])),
        "balance_classes": draw(st.booleans()),
        "knn_weights": draw(st.sampled_from(["uniform", "distance"])),
    }
    permutation = rng.permutation(n_samples)
    return X, y, params, permutation


@SETTINGS
@given(problems())
def test_weights_are_vote_shares_within_the_budget(problem):
    X, y, params, _ = problem
    clf = SubspaceKNNClassifier(**params).fit(X, y)
    n_votes = len(clf.selection_path_)
    assert_allclose(clf.subspace_weights_.sum(), 1.0)
    assert_allclose(clf.subspace_weights_ * n_votes, np.round(clf.subspace_weights_ * n_votes))
    assert params["n_subspaces"] is None or len(clf.subspaces_) <= params["n_subspaces"]
    assert {subspace for subspace, _ in clf.selection_path_} == set(clf.subspaces_)


@SETTINGS
@given(problems())
def test_predict_is_the_argmax_of_predict_proba(problem):
    X, y, params, _ = problem
    clf = SubspaceKNNClassifier(**params).fit(X, y)
    proba = clf.predict_proba(X)
    assert_allclose(proba.sum(axis=1), 1.0)
    assert_array_equal(clf.predict(X), clf.classes_[np.argmax(proba, axis=1)])


@SETTINGS
@given(problems())
def test_selection_is_invariant_to_row_order(problem):
    X, y, params, permutation = problem
    first = SubspaceKNNClassifier(**params).fit(X, y)
    second = SubspaceKNNClassifier(**params).fit(X[permutation], y[permutation])
    assert second.subspaces_ == first.subspaces_
    assert_array_equal(second.subspace_weights_, first.subspace_weights_)
    assert_allclose(second.predict_proba(X), first.predict_proba(X), atol=1e-12)
