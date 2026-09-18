"""scikit-learn's own estimator contract, run against two configurations."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from subspaceknn import SubspaceKNNClassifier


@parametrize_with_checks(
    [
        SubspaceKNNClassifier(),
        SubspaceKNNClassifier(
            subspace_size=(1, 2),
            n_subspaces=None,
            voting="hard",
            weighting="uniform",
            cv=3,
        ),
    ],
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
