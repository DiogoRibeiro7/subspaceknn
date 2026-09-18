"""scikit-learn's own estimator contract, run against three configurations."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from subspaceknn import SubspaceKNNClassifier


@parametrize_with_checks(
    [
        SubspaceKNNClassifier(),
        SubspaceKNNClassifier(
            subspace_size=(1, 2),
            n_subspaces=None,
            voting="hard",
            balance_classes=False,
            knn_weights="distance",
            cv=3,
        ),
        SubspaceKNNClassifier(
            subspace_size=(1, 2),
            n_subspaces=None,
            selection="ranked",
            voting="hard",
            weighting="uniform",
            cv=3,
        ),
    ],
)
def test_sklearn_compatible(estimator, check):
    check(estimator)
