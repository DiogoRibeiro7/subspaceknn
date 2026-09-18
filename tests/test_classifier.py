import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from subspaceknn import SubspaceKNNClassifier
from subspaceknn._classifier import _complementary_selection


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
    clf = SubspaceKNNClassifier(subspace_size=2, n_subspaces=None, selection="ranked").fit(X, y)
    assert clf.candidate_subspaces_ == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(clf.subspaces_) == 6
    assert list(clf.subspace_scores_) == sorted(clf.subspace_scores_, reverse=True)
    assert_allclose(clf.subspace_weights_.sum(), 1.0)
    assert len(clf.estimators_) == 6


def test_n_subspaces_limits_the_ranked_ensemble(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=2, n_subspaces=2, selection="ranked").fit(X, y)
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
    clf = SubspaceKNNClassifier(selection="ranked", weighting="uniform", n_subspaces=4).fit(X, y)
    assert_allclose(clf.subspace_weights_, 0.25)


def test_score_weighting_is_proportional_to_scores(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(selection="ranked", weighting="score", n_subspaces=None).fit(X, y)
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
        {"selection": "best"},
        {"max_votes": 0},
        {"balance_classes": "yes"},
        {"cv": 1},
        {"cv": "kfold"},
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


def test_non_partition_splitter_is_rejected(iris):
    X, y = iris
    splitter = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    with pytest.raises(ValueError, match="partition"):
        SubspaceKNNClassifier(cv=splitter).fit(X, y)


# ------------------------------------------------------- complementary selection


def _crafted_votes():
    """Three candidates on ten samples of class 0 or 1.

    Candidates 0 and 1 are identical: confident and right on samples 0-4, unsure
    on samples 5-9. Candidate 2 is unsure on samples 0-4 and nearly right on
    samples 5-9, so it is a little worse alone but covers what the others miss.
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    onehot = np.eye(2)[y]
    unsure = np.full((5, 2), 0.5)
    strong = np.vstack([onehot[:5], unsure])
    other = np.vstack([unsure, 0.9 * onehot[5:] + 0.05])
    return np.stack([strong, strong.copy(), other]), y


def test_complementary_selection_skips_duplicates_for_complementary_votes():
    votes, y = _crafted_votes()
    order, weights, path = _complementary_selection(
        votes, y, np.ones(len(y)), budget=None, max_votes=10
    )
    assert order[0] == 0
    assert 2 in order
    assert 1 not in order
    assert_allclose(weights.sum(), 1.0)
    assert path[0][0] == 0
    assert path[-1][1] == min(loss for _, loss in path)


def test_complementary_selection_respects_the_budget():
    votes, y = _crafted_votes()
    order, weights, path = _complementary_selection(
        votes, y, np.ones(len(y)), budget=1, max_votes=10
    )
    assert order.tolist() == [0]
    assert_allclose(weights, [1.0])
    assert len(path) == 1


def test_balanced_selection_serves_the_minority_class():
    # Nine samples of class 0 and one of class 1. Candidate 0 is right on the
    # majority and wrong on the minority; candidate 1 is unsure on the majority
    # and right on the minority.
    y = np.array([0] * 9 + [1])
    majority = np.tile([0.9, 0.1], (10, 1))
    minority = np.vstack([np.tile([0.5, 0.5], (9, 1)), [[0.0, 1.0]]])
    votes = np.stack([majority, minority])
    plain, _, _ = _complementary_selection(votes, y, np.ones(10), budget=1, max_votes=5)
    balanced_weight = np.where(y == 1, 5.0, 10 / 18)
    balanced, _, _ = _complementary_selection(votes, y, balanced_weight, budget=1, max_votes=5)
    assert plain.tolist() == [0]
    assert balanced.tolist() == [1]


def test_complementary_weights_are_vote_counts(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(subspace_size=(1, 2), max_votes=20).fit(X, y)
    n_votes = len(clf.selection_path_)
    assert 1 <= n_votes <= 20
    assert_allclose(clf.subspace_weights_ * n_votes, np.round(clf.subspace_weights_ * n_votes))
    assert_allclose(clf.subspace_weights_.sum(), 1.0)
    assert list(clf.subspace_weights_) == sorted(clf.subspace_weights_, reverse=True)
    assert {subspace for subspace, _ in clf.selection_path_} == set(clf.subspaces_)
    losses = [loss for _, loss in clf.selection_path_]
    assert losses[-1] == min(losses)


def test_n_subspaces_is_a_budget_of_distinct_subspaces(iris):
    X, y = iris
    for budget in (1, 2, 3):
        clf = SubspaceKNNClassifier(subspace_size=(1, 2), n_subspaces=budget).fit(X, y)
        assert 1 <= len(clf.subspaces_) <= budget
        assert len(set(clf.subspaces_)) == len(clf.subspaces_)


def test_ranked_selection_has_no_selection_path(iris):
    X, y = iris
    clf = SubspaceKNNClassifier(selection="ranked").fit(X, y)
    assert clf.selection_path_ is None


def test_complementary_selection_avoids_redundant_subspaces():
    # The class is positive when either of two signals is high. Features 0-2 are
    # near copies of the first signal, feature 3 is the second, which is slightly
    # weaker on its own. Ranked selection keeps two copies of the first signal;
    # complementary selection pairs it with the second, which is right exactly on
    # the positives the first signal cannot see.
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=(2, 400))
    y = ((first > 0.4) | (second > 0.6)).astype(int)
    copies = first[:, None] + 0.01 * rng.normal(size=(400, 3))
    X = np.column_stack([copies, second])
    complementary = SubspaceKNNClassifier(subspace_size=1, n_subspaces=2).fit(X, y)
    assert (3,) in complementary.subspaces_
    ranked = SubspaceKNNClassifier(subspace_size=1, n_subspaces=2, selection="ranked").fit(X, y)
    assert (3,) not in ranked.subspaces_


# ---------------------------------------------------------- out-of-fold scores


@pytest.mark.parametrize("knn_weights", ["uniform", "distance"])
def test_leave_one_out_equals_refitting_without_each_sample(knn_weights):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.integers(0, 3, size=30)
    clf = SubspaceKNNClassifier(knn_weights=knn_weights, n_neighbors=4)
    clf.classes_ = np.arange(3)
    fast = clf._leave_one_out_proba(X, y, 3)
    for row in range(30):
        model = KNeighborsClassifier(n_neighbors=4, weights=knn_weights)
        model.fit(np.delete(X, row, axis=0), np.delete(y, row))
        assert_allclose(fast[row], model.predict_proba(X[row : row + 1])[0])


def test_leave_one_out_gives_duplicates_all_the_distance_weight():
    X = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 0, 0])
    clf = SubspaceKNNClassifier(knn_weights="distance", n_neighbors=3)
    clf.classes_ = np.arange(2)
    proba = clf._leave_one_out_proba(X, y, 2)
    assert_allclose(proba[0], [0.0, 1.0])


def test_candidate_scores_are_leave_one_out_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    clf = SubspaceKNNClassifier(subspace_size=1, scoring="accuracy").fit(X, y)
    for subspace, score in zip(clf.candidate_subspaces_, clf.candidate_scores_, strict=True):
        column = X[:, list(subspace)]
        hits = 0
        for row in range(len(y)):
            model = KNeighborsClassifier().fit(np.delete(column, row, axis=0), np.delete(y, row))
            hits += int(model.predict(column[row : row + 1])[0] == y[row])
        assert score == pytest.approx(hits / len(y))


@pytest.mark.parametrize("scoring", ["accuracy", "roc_auc", "neg_log_loss", "balanced_accuracy"])
def test_any_scorer_is_evaluated_on_out_of_fold_predictions(scoring):
    X, y = make_classification(n_samples=120, n_features=4, random_state=3)
    clf = SubspaceKNNClassifier(scoring=scoring).fit(X, y)
    assert np.isfinite(clf.candidate_scores_).all()


def test_leave_one_out_falls_back_to_resubstitution_when_too_small():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.1, 5.2]])
    y = np.array([0, 0, 1, 1])
    clf = SubspaceKNNClassifier(n_neighbors=4).fit(X, y)
    assert np.isfinite(clf.candidate_scores_).all()
