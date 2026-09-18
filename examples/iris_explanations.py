"""Fit on iris, explain one prediction, and save the subspace plots.

Run with: uv run python examples/iris_explanations.py
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from subspaceknn import SubspaceKNNClassifier
from subspaceknn.plotting import plot_subspaces

mpl.use("Agg")


def main() -> None:
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target_names[iris.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    clf = SubspaceKNNClassifier(subspace_size=(1, 2), n_subspaces=4).fit(X_train, y_train)
    print(f"test accuracy: {clf.score(X_test, y_test):.3f}")
    print("subspaces used for prediction:")
    for subspace, score, weight in zip(
        clf.subspaces_, clf.subspace_scores_, clf.subspace_weights_, strict=True
    ):
        names = ", ".join(X.columns[list(subspace)])
        print(f"  {names:<40s} score {score:.3f}   weight {weight:.3f}")

    print("\nwhen each subspace joined, with the out-of-fold loss after that vote:")
    joined = set()
    for vote, (subspace, loss) in enumerate(clf.selection_path_, start=1):
        if subspace not in joined:
            joined.add(subspace)
            names = ", ".join(X.columns[list(subspace)])
            print(f"  vote {vote:>2d}: {names:<38s} loss {loss:.4f}")
    final_loss = clf.selection_path_[-1][1]
    print(f"  best ensemble after {len(clf.selection_path_)} votes, loss {final_loss:.4f}")

    explanation = clf.explain(X_test.iloc[:1])[0]
    print(
        f"\nfirst test sample: true={y_test[0]} predicted={explanation.prediction} "
        f"agreement={explanation.agreement():.2f}"
    )
    print(pd.DataFrame.from_records(explanation.to_records()).to_string(index=False))

    output = Path(__file__).with_name("output")
    output.mkdir(exist_ok=True)
    fig = plot_subspaces(clf, X_train, y_train, sample=X_test.iloc[0].to_numpy())
    target = output / "iris_subspaces.png"
    fig.savefig(target, dpi=120)
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
