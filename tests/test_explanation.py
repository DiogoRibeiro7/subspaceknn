import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris

from subspaceknn import Explanation, SubspaceKNNClassifier, SubspaceVote


@pytest.fixture(scope="module")
def fitted():
    X, y = load_iris(return_X_y=True)
    return SubspaceKNNClassifier(subspace_size=(1, 2), n_subspaces=4).fit(X, y), X, y


def test_one_explanation_per_sample(fitted):
    clf, X, _ = fitted
    explanations = clf.explain(X[:7])
    assert len(explanations) == 7
    assert all(isinstance(explanation, Explanation) for explanation in explanations)
    assert all(len(explanation.votes) == 4 for explanation in explanations)


def test_votes_are_ordered_by_weight_and_match_the_ensemble(fitted):
    clf, X, _ = fitted
    proba = clf.predict_proba(X[:10])
    predictions = clf.predict(X[:10])
    for row, explanation in enumerate(clf.explain(X[:10])):
        weights = [vote.weight for vote in explanation.votes]
        assert weights == sorted(weights, reverse=True)
        assert_allclose(sum(weights), 1.0)
        assert_allclose(explanation.probabilities, proba[row])
        assert explanation.prediction == predictions[row]
        assert_array_equal(explanation.classes, clf.classes_)
        for vote in explanation.votes:
            assert isinstance(vote, SubspaceVote)
            assert_allclose(vote.probabilities.sum(), 1.0)
            assert vote.prediction == clf.classes_[int(np.argmax(vote.probabilities))]
            assert vote.features in clf.subspaces_
            assert len(vote.feature_names) == len(vote.features)


def test_agreement_is_the_weight_of_concurring_votes(fitted):
    clf, X, _ = fitted
    for explanation in clf.explain(X[::15]):
        expected = sum(
            vote.weight for vote in explanation.votes if vote.prediction == explanation.prediction
        )
        assert explanation.agreement() == pytest.approx(expected)
        assert 0.0 <= explanation.agreement() <= 1.0


def test_unanimous_hard_votes_give_full_agreement():
    X, y = load_iris(return_X_y=True)
    clf = SubspaceKNNClassifier(voting="hard").fit(X, y)
    # The first iris sample is a textbook setosa; every subspace should agree.
    explanation = clf.explain(X[:1])[0]
    assert explanation.agreement() == pytest.approx(1.0)


def test_records_are_dataframe_ready(fitted):
    clf, X, _ = fitted
    records = clf.explain(X[:1])[0].to_records()
    assert len(records) == 4
    expected_keys = {"features", "score", "weight", "prediction", "agrees", "p(0)", "p(1)", "p(2)"}
    assert all(set(record) == expected_keys for record in records)
    assert all(isinstance(record["agrees"], bool) for record in records)
