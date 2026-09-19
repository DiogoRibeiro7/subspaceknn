"""The fitted model is the same on every platform.

Each case fits a model and compares it with values committed in
``tests/fixtures/reproducibility.json``. CI runs this module on Linux, macOS and
Windows, so a platform that selects different subspaces, weights them
differently or predicts differently fails here. The selected subspaces and
weights must match exactly; probabilities, which pass through a BLAS dot product,
to 1e-12.

When the method changes on purpose, regenerate the fixture with
``uv run python tests/test_reproducibility.py`` and explain the change in the
changelog.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification

from subspaceknn import SubspaceKNNClassifier

FIXTURE = Path(__file__).with_name("fixtures") / "reproducibility.json"
N_PROBABILITY_ROWS = 20


def _rounded_synthetic():
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5, n_redundant=1, random_state=0
    )
    return np.round(X, 1), y


CASES = {
    "iris": (lambda: load_iris(return_X_y=True), {}),
    # The configuration whose selection differed between macOS and Linux in 0.2.0.
    "iris-sizes-1-2": (
        lambda: load_iris(return_X_y=True),
        {"subspace_size": (1, 2), "n_subspaces": 4},
    ),
    "wine-hard-distance": (
        lambda: load_wine(return_X_y=True),
        {"voting": "hard", "knn_weights": "distance"},
    ),
    "breast-cancer": (lambda: load_breast_cancer(return_X_y=True), {}),
    "rounded-sizes-1-3": (_rounded_synthetic, {"subspace_size": (1, 2, 3), "n_subspaces": 8}),
}


def fit_case(name):
    load, params = CASES[name]
    X, y = load()
    clf = SubspaceKNNClassifier(**params).fit(X, y)
    return {
        "subspaces": [list(subspace) for subspace in clf.subspaces_],
        "weights": clf.subspace_weights_.tolist(),
        "probabilities": clf.predict_proba(X[:N_PROBABILITY_ROWS]).tolist(),
    }


@pytest.fixture(scope="module")
def expected():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(CASES))
def test_fit_matches_the_committed_fixture(name, expected):
    actual = fit_case(name)
    assert actual["subspaces"] == expected[name]["subspaces"]
    assert_array_equal(actual["weights"], expected[name]["weights"])
    assert_allclose(actual["probabilities"], expected[name]["probabilities"], rtol=0, atol=1e-12)


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    values = {name: fit_case(name) for name in CASES}
    FIXTURE.write_text(json.dumps(values, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE}")
