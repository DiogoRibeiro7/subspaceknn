import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from subspaceknn import SubspaceKNNClassifier


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


def test_iris_accuracy_is_competitive_with_plain_knn(iris):
    X, y = iris
    ours = cross_val_score(SubspaceKNNClassifier(), X, y, cv=5, scoring="f1_macro").mean()
    knn = cross_val_score(KNeighborsClassifier(), X, y, cv=5, scoring="f1_macro").mean()
    assert ours > 0.9
    assert ours >= knn - 0.03


def test_probabilities_are_normalised_and_consistent_with_predict(iris):
    X, y = iris
    clf = SubspaceKNNClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3)
    assert_allclose(proba.sum(axis=1), 1.0)
    assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(X))


def test_candidates_are_enumerated_in_lexicographic_order(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=2, n_subspaces=None).fit(X, y)
    assert clf.candidate_subspaces_ == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(clf.subspaces_) == 6
    assert list(clf.subspace_scores_) == sorted(clf.subspace_scores_, reverse=True)
    assert_allclose(clf.subspace_weights_.sum(), 1.0)
    assert len(clf.estimators_) == 6


def test_n_subspaces_limits_the_ensemble(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=2, n_subspaces=2).fit(X, y)
    assert len(clf.subspaces_) == 2
    assert len(clf.candidate_subspaces_) == 6
    best = int(np.argmax(clf.candidate_scores_))
    assert clf.subspaces_[0] == clf.candidate_subspaces_[best]


def test_mixed_sizes_enumerate_every_size(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=(3, 1), n_subspaces=None).fit(X, y)
    sizes = [len(subspace) for subspace in clf.candidate_subspaces_]
    assert sizes == [1, 1, 1, 1, 3, 3, 3, 3]


def test_sizes_beyond_n_features_are_ignored(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=(2, 10)).fit(X, y)
    assert all(len(subspace) == 2 for subspace in clf.candidate_subspaces_)


def test_all_sizes_too_large_raises_with_feature_count(iris):
    X, y = iris
    with pytest.raises(ValueError, match="n_features=4"):
        SubspaceKNNClassifier(subspace_size=5).fit(X, y)


def test_screening_keeps_candidates_within_the_cap():
    X, y = make_classification(n_samples=200, n_features=12, n_informative=4, random_state=0)
    clf = SubspaceKNNClassifier(subspace_size=2, max_candidates=10).fit(X, y)
    assert len(clf.candidate_subspaces_) == 10
    assert len(clf.screened_features_) == 5
    assert clf.feature_screening_scores_.shape == (12,)
    kept = set(clf.screened_features_.tolist())
    assert all(feature in kept for subspace in clf.candidate_subspaces_ for feature in subspace)
    ranked = np.argsort(-clf.feature_screening_scores_, kind="stable")[:5]
    assert kept == set(ranked.tolist())


def test_screening_with_mixed_sizes_respects_the_cap():
    X, y = make_classification(n_samples=200, n_features=12, n_informative=4, random_state=1)
    clf = SubspaceKNNClassifier(subspace_size=(1, 2, 3), max_candidates=50).fit(X, y)
    # 6 features give 6 + 15 + 20 = 41 candidates; 7 would give 63.
    assert len(clf.screened_features_) == 6
    assert len(clf.candidate_subspaces_) == 41


def test_no_screening_when_within_the_cap(iris):
    X, y = iris
    clf = SubspaceKNNClassifier().fit(X, y)
    assert clf.feature_screening_scores_ is None
    assert_array_equal(clf.screened_features_, [0, 1, 2, 3])


def test_max_candidates_none_evaluates_everything():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=3, random_state=2)
    clf = SubspaceKNNClassifier(subspace_size=2, max_candidates=None).fit(X, y)
    assert len(clf.candidate_subspaces_) == 28


def test_uniform_weighting_gives_equal_weights(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(weighting="uniform", n_subspaces=4).fit(X, y)
    assert_allclose(clf.subspace_weights_, 0.25)


def test_score_weighting_is_proportional_to_scores(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(weighting="score", n_subspaces=None).fit(X, y)
    expected = clf.subspace_scores_ / clf.subspace_scores_.sum()
    assert_allclose(clf.subspace_weights_, expected)


def test_hard_voting_uses_one_hot_votes(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(voting="hard").fit(X, y)
    for explanation in clf.explain(X[:5]):
        for vote in explanation.votes:
            assert sorted(vote.probabilities) == [0.0, 0.0, 1.0]


def test_feature_scores_are_zero_for_unused_features(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=2, n_subspaces=1).fit(X, y)
    used = clf.subspaces_[0]
    for feature in range(4):
        if feature in used:
            assert clf.feature_scores_[feature] == pytest.approx(clf.subspace_scores_[0])
        else:
            assert clf.feature_scores_[feature] == 0.0


def test_dataframe_input_and_string_labels(iris):
    X, y = iris
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    frame = pd.DataFrame(X, columns=names)
    labels = np.array(["setosa", "versicolor", "virginica"])[y]
    clf = SubspaceKNNClassifier().fit(frame, labels)
    assert_array_equal(clf.feature_names_in_, names)
    assert_array_equal(clf.classes_, ["setosa", "versicolor", "virginica"])
    predictions = clf.predict(frame)
    assert set(predictions) <= set(labels)
    explanation = clf.explain(frame.iloc[:1])[0]
    assert all(name in names for vote in explanation.votes for name in vote.feature_names)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_neighbors": 0},
        {"n_neighbors": 2.5},
        {"subspace_size": 0},
        {"subspace_size": []},
        {"subspace_size": "two"},
        {"n_subspaces": 0},
        {"max_candidates": 0},
        {"voting": "loud"},
        {"weighting": "random"},
        {"cv": 1},
    ],
)
def test_invalid_hyperparameters_raise_in_fit(iris, kwargs):
    X, y = iris
    with pytest.raises(ValueError, match="must be"):
        SubspaceKNNClassifier(**kwargs).fit(X, y)


def test_unfitted_estimator_raises(iris):
    X, _ = iris
    with pytest.raises(NotFittedError):
        SubspaceKNNClassifier().predict(X)


def test_clone_and_grid_search(iris):
    X, y = iris
    search = GridSearchCV(
        SubspaceKNNClassifier(cv=3),
        {"n_neighbors": [3, 7], "voting": ["soft", "hard"]},
        cv=3,
        scoring="f1_macro",
    ).fit(X, y)
    assert search.best_score_ > 0.9
    assert clone(search.best_estimator_).get_params() == search.best_estimator_.get_params()


def test_tiny_dataset_falls_back_to_resubstitution_scores():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 5.0], [5.0, 5.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = SubspaceKNNClassifier(n_neighbors=2, cv=5, subspace_size=(1, 2)).fit(X, y)
    assert np.isfinite(clf.candidate_scores_).all()
    assert_array_equal(clf.predict(X), y)


def test_single_class_fits_and_predicts_that_class():
    X = np.random.default_rng(0).normal(size=(12, 3))
    y = np.full(12, "only")
    clf = SubspaceKNNClassifier().fit(X, y)
    assert_array_equal(clf.predict(X), y)
    assert_allclose(clf.predict_proba(X), 1.0)


def test_more_neighbours_than_samples_raises():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="n_samples=3"):
        SubspaceKNNClassifier(n_neighbors=5).fit(X, y)


def test_custom_cross_validator_is_used(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(cv=KFold(n_splits=4, shuffle=True, random_state=0)).fit(X, y)
    assert len(clf.candidate_scores_) == 6
    assert np.isfinite(clf.candidate_scores_).all()
