"""Regenerate the figures used by the documentation site.

Run with: uv run python scripts/docs_figures.py
"""

from pathlib import Path

import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from subspaceknn import SubspaceKNNClassifier
from subspaceknn.plotting import plot_subspaces

mpl.use("Agg")

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"


def main() -> None:
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target_names[iris.target]
    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0, stratify=y)
    clf = SubspaceKNNClassifier(subspace_size=(1, 2), n_subspaces=4).fit(X_train, y_train)

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig = plot_subspaces(clf, X_train, y_train, sample=X_test.iloc[0].to_numpy(), panel_size=3.6)
    target = ASSETS / "iris-subspaces.png"
    fig.savefig(target, dpi=110, bbox_inches="tight", metadata={"Software": None})
    print(f"saved {target}")


if __name__ == "__main__":
    main()
