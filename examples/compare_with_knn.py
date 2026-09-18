"""Cross-validated macro-F1 of subspace ensembles against plain kNN on toy datasets.

Run with: uv run python examples/compare_with_knn.py
"""

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from subspaceknn import SubspaceKNNClassifier

DATASETS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
MODELS = {
    "kNN": KNeighborsClassifier(),
    "pairs": SubspaceKNNClassifier(subspace_size=2),
    "sizes 1-3": SubspaceKNNClassifier(subspace_size=(1, 2, 3), n_subspaces=8),
}


def main() -> None:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    header = f"{'dataset':<14s}" + "".join(f"{name:>12s}" for name in MODELS)
    print(header)
    for dataset, loader in DATASETS.items():
        X, y = loader(return_X_y=True)
        row = f"{dataset:<14s}"
        for model in MODELS.values():
            pipeline = make_pipeline(StandardScaler(), model)
            score = cross_val_score(pipeline, X, y, cv=folds, scoring="f1_macro").mean()
            row += f"{score:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
