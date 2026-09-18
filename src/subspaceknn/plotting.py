"""Visualise the subspaces of a fitted :class:`~subspaceknn.SubspaceKNNClassifier`.

This module needs matplotlib, which is an optional dependency::

    pip install "subspaceknn[plot]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import ArrayLike, NDArray
    from sklearn.neighbors import KNeighborsClassifier

    from subspaceknn._classifier import SubspaceKNNClassifier

_PALETTE = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def plot_subspaces(
    estimator: SubspaceKNNClassifier,
    X: ArrayLike,
    y: ArrayLike,
    *,
    sample: ArrayLike | None = None,
    n_subspaces: int | None = None,
    grid_resolution: int = 100,
    panel_size: float = 4.0,
) -> Figure:
    """Draw the training data in the best subspaces, with decision regions where possible.

    Parameters
    ----------
    estimator : SubspaceKNNClassifier
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Data to draw, usually the training data.
    y : array-like of shape (n_samples,)
        Class labels of ``X``.
    sample : array-like of shape (n_features,), optional
        A single sample to highlight with a star in every panel, for example a
        test point whose prediction is being explained.
    n_subspaces : int, optional
        Number of panels, from the best subspace onwards. Defaults to all
        subspaces used for prediction.
    grid_resolution : int, default=100
        Number of grid points per axis for the decision regions of one- and
        two-dimensional subspaces.
    panel_size : float, default=4.0
        Width and height of each panel in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; it is not shown or saved.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "plot_subspaces needs matplotlib; install it with 'pip install subspaceknn[plot]'.",
        ) from error

    check_is_fitted(estimator)
    X_arr = validate_data(estimator, X, reset=False, dtype="numeric")
    y_arr = np.asarray(y)
    sample_arr = None if sample is None else np.asarray(sample, dtype=np.float64).reshape(-1)
    names = estimator._feature_names()  # noqa: SLF001
    subspaces = estimator.subspaces_ if n_subspaces is None else estimator.subspaces_[:n_subspaces]
    n_panels = max(len(subspaces), 1)

    fig = plt.figure(figsize=(panel_size * n_panels, panel_size))
    for panel, subspace in enumerate(subspaces, start=1):
        estimator_index = estimator.subspaces_.index(subspace)
        model = estimator.estimators_[estimator_index]
        score = float(estimator.subspace_scores_[estimator_index])
        weight = float(estimator.subspace_weights_[estimator_index])
        title = (
            f"{', '.join(names[index] for index in subspace)}\n"
            f"weight {weight:.2f}, score {score:.3f}"
        )
        if len(subspace) == 1:
            ax = fig.add_subplot(1, n_panels, panel)
            _draw_1d(
                ax,
                model=model,
                classes=estimator.classes_,
                X=X_arr,
                y=y_arr,
                subspace=subspace,
                sample=sample_arr,
                grid_resolution=grid_resolution,
            )
        elif len(subspace) == 2:  # noqa: PLR2004
            ax = fig.add_subplot(1, n_panels, panel)
            _draw_2d(
                ax,
                model=model,
                classes=estimator.classes_,
                X=X_arr,
                y=y_arr,
                subspace=subspace,
                sample=sample_arr,
                grid_resolution=grid_resolution,
            )
        elif len(subspace) == 3:  # noqa: PLR2004
            axes3d = cast("Axes3D", fig.add_subplot(1, n_panels, panel, projection="3d"))
            _draw_3d(
                axes3d,
                classes=estimator.classes_,
                X=X_arr,
                y=y_arr,
                subspace=subspace,
                sample=sample_arr,
            )
            ax = axes3d
        else:
            ax = fig.add_subplot(1, n_panels, panel)
            _draw_2d(
                ax,
                model=None,
                classes=estimator.classes_,
                X=X_arr,
                y=y_arr,
                subspace=subspace[:2],
                sample=sample_arr,
                grid_resolution=grid_resolution,
            )
            title += f" (first two of {len(subspace)} features)"
        ax.set_title(title)
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def _colour(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


def _draw_1d(
    ax: Axes,
    *,
    model: KNeighborsClassifier,
    classes: NDArray[Any],
    X: NDArray[np.float64],
    y: NDArray[Any],
    subspace: Sequence[int],
    sample: NDArray[np.float64] | None,
    grid_resolution: int,
) -> None:
    column = X[:, subspace[0]]
    low, high = _padded_range(column)
    grid = np.linspace(low, high, grid_resolution)
    regions = model.predict(grid.reshape(-1, 1))
    for index in range(len(grid) - 1):
        ax.axvspan(
            grid[index], grid[index + 1], color=_colour(int(regions[index])), alpha=0.12, lw=0
        )
    for class_index, label in enumerate(classes):
        mask = y == label
        ax.scatter(
            column[mask],
            np.full(mask.sum(), class_index),
            s=18,
            alpha=0.6,
            c=_colour(class_index),
            label=str(label),
        )
    if sample is not None:
        ax.axvline(sample[subspace[0]], color="black", lw=1.2, ls="--", label="sample")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([str(label) for label in classes])
    ax.set_xlim(low, high)


def _draw_2d(
    ax: Axes,
    *,
    model: KNeighborsClassifier | None,
    classes: NDArray[Any],
    X: NDArray[np.float64],
    y: NDArray[Any],
    subspace: Sequence[int],
    sample: NDArray[np.float64] | None,
    grid_resolution: int,
) -> None:
    first, second = subspace[0], subspace[1]
    x_low, x_high = _padded_range(X[:, first])
    y_low, y_high = _padded_range(X[:, second])
    if model is not None:
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(x_low, x_high, grid_resolution),
            np.linspace(y_low, y_high, grid_resolution),
        )
        grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        regions = model.predict(grid).reshape(mesh_x.shape)
        levels = np.arange(len(classes) + 1) - 0.5
        colours = [_colour(index) for index in range(len(classes))]
        ax.contourf(mesh_x, mesh_y, regions, levels=levels, colors=colours, alpha=0.12)
    for class_index, label in enumerate(classes):
        mask = y == label
        ax.scatter(
            X[mask, first],
            X[mask, second],
            s=18,
            alpha=0.6,
            c=_colour(class_index),
            label=str(label),
        )
    if sample is not None:
        ax.plot(
            sample[first], sample[second], marker="*", markersize=16, color="black", label="sample"
        )
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)


def _draw_3d(
    ax: Axes3D,
    *,
    classes: NDArray[Any],
    X: NDArray[np.float64],
    y: NDArray[Any],
    subspace: Sequence[int],
    sample: NDArray[np.float64] | None,
) -> None:
    first, second, third = subspace[0], subspace[1], subspace[2]
    for class_index, label in enumerate(classes):
        mask = y == label
        ax.scatter(
            X[mask, first],
            X[mask, second],
            X[mask, third],
            s=14,
            alpha=0.6,
            c=_colour(class_index),
            label=str(label),
        )
    if sample is not None:
        ax.scatter(
            [sample[first]],
            [sample[second]],
            [sample[third]],
            marker="*",
            s=200,
            c="black",
            label="sample",
        )


def _padded_range(values: NDArray[np.float64]) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.05 * (high - low) if high > low else 0.5
    return low - pad, high + pad
