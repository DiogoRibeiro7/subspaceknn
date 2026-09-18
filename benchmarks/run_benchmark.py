"""Benchmark of subspace selection schemes against plain kNN.

Reproduces the tables in docs/benchmark.md. Downloads eleven datasets from OpenML
on first use (cached by scikit-learn afterwards), so it needs network access once.

Run with: uv run python benchmarks/run_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from subspaceknn import SubspaceKNNClassifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator

# OpenML data ids, pinned so that the benchmark does not drift with new versions.
OPENML = {
    "diabetes": 37,
    "banknote": 1462,
    "ionosphere": 59,
    "sonar": 40,
    "vehicle": 54,
    "glass": 41,
    "blood-transfusion": 1464,
    "ilpd": 1480,
    "climate-crashes": 1467,
    "segment": 36,
    "qsar-biodeg": 1494,
}
BUILTIN: dict[str, Callable[..., tuple[NDArray[np.float64], NDArray[np.int_]]]] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast-cancer": load_breast_cancer,
}
SETTINGS = {
    "pairs, 5 subspaces": {"subspace_size": 2, "n_subspaces": 5},
    "pairs, 3 subspaces": {"subspace_size": 2, "n_subspaces": 3},
    "sizes 1-3, 8 subspaces": {"subspace_size": (1, 2, 3), "n_subspaces": 8},
}
# The 0.1.0 configuration: ikNN-style ranking scored by 5-fold CV over at most 100 candidates.
METHODS = {
    "ranked (0.1.0)": {"selection": "ranked", "cv": 5, "max_candidates": 100},
    "ranked, larger pool": {"selection": "ranked"},
    "complementary": {},
}
THRESHOLD = 0.005


@dataclass
class Row:
    dataset: str
    shape: tuple[int, int]
    scores: dict[str, float]
    fit_seconds: float


def load(name: str) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    if name in BUILTIN:
        X, y = BUILTIN[name](return_X_y=True)
        return np.asarray(X, dtype=np.float64), np.asarray(y)
    bunch = fetch_openml(data_id=OPENML[name], as_frame=True, parser="auto")
    X = bunch.data.select_dtypes("number").to_numpy(dtype=np.float64)
    return X, LabelEncoder().fit_transform(np.asarray(bunch.target))


def macro_f1(
    model: BaseEstimator, X: NDArray[np.float64], y: NDArray[np.int_], repeats: int
) -> float:
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=0)
    pipeline = make_pipeline(StandardScaler(), model)
    return float(cross_val_score(pipeline, X, y, cv=folds, scoring="f1_macro", n_jobs=-1).mean())


def run(setting: dict[str, object], names: list[str], repeats: int) -> list[Row]:
    rows = []
    for name in names:
        X, y = load(name)
        scores = {"kNN": macro_f1(KNeighborsClassifier(), X, y, repeats)}
        for method, params in METHODS.items():
            model = SubspaceKNNClassifier(**setting, **params)  # type: ignore[arg-type]
            scores[method] = macro_f1(model, X, y, repeats)
        # One single-threaded fit of the default method on the whole dataset.
        X_scaled = StandardScaler().fit_transform(X)
        start = time.perf_counter()
        SubspaceKNNClassifier(**setting).fit(X_scaled, y)  # type: ignore[arg-type]
        rows.append(Row(name, X.shape, scores, time.perf_counter() - start))
        print(f"  {name:<18s} " + " ".join(f"{v:.3f}" for v in scores.values()), flush=True)
    return rows


def table(title: str, rows: list[Row]) -> str:
    columns = ["kNN", *METHODS]
    lines = [
        f"### {title}",
        "",
        "| Dataset | n x p | " + " | ".join(columns) + " | fit (s) |",
        "| --- | --- | " + " | ".join("---:" for _ in columns) + " | ---: |",
    ]
    for row in rows:
        best = max(row.scores.values())
        cells = [
            f"**{row.scores[c]:.3f}**" if row.scores[c] == best else f"{row.scores[c]:.3f}"
            for c in columns
        ]
        lines.append(
            f"| {row.dataset} | {row.shape[0]} x {row.shape[1]} | "
            + " | ".join(cells)
            + f" | {row.fit_seconds:.2f} |",
        )
    means = [np.mean([row.scores[c] for row in rows]) for c in columns]
    lines.append("| **mean** | | " + " | ".join(f"{m:.3f}" for m in means) + " | |")
    lines.append("")
    for reference in ("ranked (0.1.0)", "kNN"):
        diff = np.array([row.scores["complementary"] - row.scores[reference] for row in rows])
        lines.append(
            f"Complementary vs {reference}: mean {diff.mean():+.3f}, better on "
            f"{(diff > THRESHOLD).sum()}, worse on {(diff < -THRESHOLD).sum()} of "
            f"{len(rows)} datasets (differences under {THRESHOLD} count as ties).",
        )
        lines.append("")
    return "\n".join(lines)


def redundancy(names: list[str]) -> str:
    """Distinct features in the five pairs of each scheme, fitted on the whole dataset."""
    lines = [
        "### Redundancy of the selected pairs",
        "",
        "| Dataset | ranked (0.1.0) | complementary |",
        "| --- | --- | --- |",
    ]
    for name in names:
        X, y = load(name)
        X_scaled = StandardScaler().fit_transform(X)
        cells = []
        for params in (METHODS["ranked (0.1.0)"], METHODS["complementary"]):
            subspaces = SubspaceKNNClassifier(**params).fit(X_scaled, y).subspaces_  # type: ignore[arg-type]
            counts = Counter(feature for subspace in subspaces for feature in subspace)
            top = counts.most_common(1)[0][1]
            cells.append(f"{len(counts)} features; top one in {top} of {len(subspaces)} pairs")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="built-in datasets, one repeat")
    args = parser.parse_args()
    names = list(BUILTIN) if args.quick else [*BUILTIN, *OPENML]
    repeats = 1 if args.quick else 3
    warnings.filterwarnings("ignore", category=UserWarning)
    output = []
    for title, setting in SETTINGS.items():
        print(title, flush=True)
        output.append(table(title, run(setting, names, repeats)))
    output.append(redundancy(names))
    print()
    print("\n".join(output))


if __name__ == "__main__":
    main()
