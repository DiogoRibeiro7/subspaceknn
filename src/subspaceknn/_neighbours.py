"""Nearest-neighbour votes that do not depend on the order of equidistant neighbours."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# Metrics whose distances are recomputed one coordinate at a time with separate
# numpy operations, which round the same way on every platform.
_SQUARED_METRICS = frozenset({"minkowski", "euclidean", "l2"})
_ABSOLUTE_METRICS = frozenset({"manhattan", "cityblock", "l1"})
_MAXIMUM_METRICS = frozenset({"chebyshev", "infinity"})

# A candidate list is complete once its farthest search distance exceeds the k-th
# exact distance by more than this relative margin, which absorbs the rounding
# differences between the neighbour search and the exact recomputation.
_COMPLETENESS_MARGIN = 1e-9


class TieSharingKNeighborsClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """k-nearest-neighbour classifier whose votes do not depend on the order of ties.

    When several training points lie exactly at the distance of the k-th
    neighbour, a plain kNN keeps whichever of them its neighbour search returns
    first, and that order depends on the platform. Here, with `d*` the k-th
    smallest distance, every point strictly closer than `d*` gets a full vote and
    the points at exactly `d*` share the remaining votes equally, so the total is
    still `n_neighbors`. With `weights="distance"` each vote is further divided by
    the point's distance, and points at distance zero, if there are any, take all
    the weight.

    Training points with identical coordinates are stored once, as a cell with a
    count per class, so repeated values cost nothing extra. For the Euclidean,
    Manhattan and Chebyshev metrics, distances are recomputed from the
    coordinates with operations that round identically on every platform, and
    ties are equalities of those distances; the neighbour search only proposes
    candidates. Other metrics use the search's own distances.

    This is the model behind every subspace of
    [`SubspaceKNNClassifier`][subspaceknn.SubspaceKNNClassifier].

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of votes per prediction.
    weights : {"uniform", "distance"}, default="uniform"
        Whether votes are equal or divided by the distance.
    metric : str, default="minkowski"
        Distance metric passed to the neighbour search.
    n_jobs : int or None, default=None
        Parallelism of the neighbour search.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: Literal["uniform", "distance"] = "uniform",
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> TieSharingKNeighborsClassifier:
        """Store the training data as cells of identical points and build the search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
            The fitted model.
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        self.classes_, labels = np.unique(np.asarray(y), return_inverse=True)
        cells, cell_of_row = _group_identical_rows(X_arr)
        n_classes = len(self.classes_)
        pairs = cell_of_row * n_classes + labels.reshape(-1)
        self._counts = np.bincount(pairs, minlength=len(cells) * n_classes).reshape(
            len(cells), n_classes
        )
        self._cells = np.ascontiguousarray(cells)
        self._pairs = pairs
        self._n_samples = len(X_arr)
        self._search = NearestNeighbors(metric=self.metric, n_jobs=self.n_jobs).fit(self._cells)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the class probabilities of each sample.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_queries, n_classes)
            Vote shares aligned with `classes_`.
        """
        return self._votes(np.ascontiguousarray(X, dtype=np.float64), None, None)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest vote share; ties go to the first class.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_queries,)
            Predicted class labels.
        """
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])

    def leave_one_out_proba(self) -> NDArray[np.float64]:
        """Return each training sample's probabilities with the sample left out.

        This equals refitting without each sample and predicting it. Samples that
        share a cell and a class have the same answer, so it is computed once per
        cell and class.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Leave-one-out vote shares aligned with `classes_`.
        """
        n_classes = len(self.classes_)
        pairs, pair_of_row = np.unique(self._pairs, return_inverse=True)
        own_cell = pairs // n_classes
        proba = self._votes(self._cells[own_cell], own_cell, pairs % n_classes)
        return np.asarray(proba[pair_of_row.reshape(-1)], dtype=np.float64)

    def _votes(
        self,
        queries: NDArray[np.float64],
        own_cell: NDArray[np.intp] | None,
        own_class: NDArray[np.intp] | None,
    ) -> NDArray[np.float64]:
        """Vote shares of `queries`, leaving one point of `own_class` out of `own_cell`."""
        k = self.n_neighbors
        n_cells = len(self._cells)
        available = self._n_samples - (own_cell is not None)
        if k > available:
            raise ValueError(
                f"Expected n_neighbors <= {available} neighbours to choose from, got {k}.",
            )
        proba = np.empty((len(queries), len(self.classes_)), dtype=np.float64)
        pending = np.arange(len(queries))
        count = min(n_cells, k + 1 + (own_cell is not None))
        while pending.size:
            rows = queries[pending]
            search_distance, cell = self._search.kneighbors(rows, n_neighbors=count)
            reduced, distance = self._exact_distances(rows, cell, search_distance)
            counts = self._counts[cell]
            if own_cell is not None and own_class is not None:
                row, column = np.nonzero(cell == own_cell[pending][:, None])
                counts[row, column, own_class[pending][row]] -= 1
            # The k-th point by distance: cumulate the cell sizes in distance order.
            # Which of several equidistant cells comes first does not matter here.
            order = np.argsort(reduced, axis=1)
            cumulative = np.cumsum(np.take_along_axis(counts.sum(axis=2), order, axis=1), axis=1)
            enough = cumulative[:, -1] >= k
            position = np.take_along_axis(
                order, np.argmax(cumulative >= k, axis=1)[:, None], axis=1
            )
            boundary = np.take_along_axis(reduced, position, axis=1)
            boundary_distance = np.take_along_axis(distance, position, axis=1)[:, 0]
            # Complete when the farthest candidate is clearly beyond the k-th distance,
            # so no cell outside the list can be tied with it.
            complete = (count >= n_cells) | (
                enough & (search_distance[:, -1] > boundary_distance * (1.0 + _COMPLETENESS_MARGIN))
            )
            if complete.all():
                proba[pending] = self._shares(cell, counts, reduced, distance, boundary)
                break
            proba[pending[complete]] = self._shares(
                cell[complete],
                counts[complete],
                reduced[complete],
                distance[complete],
                boundary[complete],
            )
            pending = pending[~complete]
            count = min(n_cells, 2 * count)
        return proba

    def _exact_distances(
        self,
        rows: NDArray[np.float64],
        cell: NDArray[np.intp],
        search_distance: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a monotone reduced distance, used for ties, and the distance itself."""
        if self.metric not in _SQUARED_METRICS | _ABSOLUTE_METRICS | _MAXIMUM_METRICS:
            return search_distance, search_distance
        differences = [self._cells[cell, j] - rows[:, j : j + 1] for j in range(rows.shape[1])]
        if self.metric in _SQUARED_METRICS:
            reduced = differences[0] * differences[0]
            for difference in differences[1:]:
                reduced = reduced + difference * difference
            return reduced, np.sqrt(reduced)
        combine = np.add if self.metric in _ABSOLUTE_METRICS else np.maximum
        magnitudes = [np.abs(difference) for difference in differences]
        reduced = magnitudes[0]
        for magnitude in magnitudes[1:]:
            reduced = combine(reduced, magnitude)
        return reduced, reduced

    def _shares(
        self,
        cell: NDArray[np.intp],
        counts: NDArray[np.int64],
        reduced: NDArray[np.float64],
        distance: NDArray[np.float64],
        boundary: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply the tie rule to complete candidate lists of cells.

        `counts` holds the number of training points of each class in each
        candidate cell. Under uniform weights the class totals are exact counts
        plus a share of the tied counts, which does not depend on any order. Under
        distance weights each cell adds its count times its weight, summed one cell
        after another in a canonical order, by distance and then cell, so that the
        rounding does not depend on the order of the training rows either.
        """
        k = self.n_neighbors
        closer = (reduced < boundary)[:, :, None]
        tied = (reduced == boundary)[:, :, None]
        closer_counts = (counts * closer).sum(axis=1)
        tied_counts = (counts * tied).sum(axis=1)
        share = (k - closer_counts.sum(axis=1, keepdims=True)) / tied_counts.sum(
            axis=1, keepdims=True
        )
        if self.weights == "uniform":
            totals = closer_counts + share * tied_counts
        else:
            per_point = np.where(closer[:, :, 0], 1.0, np.where(tied[:, :, 0], share, 0.0))
            occupied = counts.sum(axis=2) > 0
            at_zero = (reduced == 0.0) & occupied
            with np.errstate(divide="ignore", invalid="ignore"):
                weighted = per_point / distance
            per_point = np.where(at_zero.any(axis=1, keepdims=True), at_zero * 1.0, weighted)
            # Empty cells, such as a sample's own cell once the sample is left out,
            # carry no weight; this also clears the infinite weight at distance zero.
            per_point = np.where(occupied & (per_point > 0.0), per_point, 0.0)
            order = np.lexsort((cell, reduced), axis=-1)
            per_point = np.take_along_axis(per_point, order, axis=1)
            counts = np.take_along_axis(counts, order[:, :, None], axis=1)
            totals = np.cumsum(counts * per_point[:, :, None], axis=1)[:, -1, :]
        return np.asarray(totals / totals.sum(axis=1, keepdims=True), dtype=np.float64)


def _group_identical_rows(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return the distinct rows of `X` in lexicographic order and each row's position among them.

    Sorting the columns and comparing neighbours is several times faster than
    `np.unique(X, axis=0)`, and the order of the distinct rows depends only on their
    coordinates, not on the order of the rows of `X`.
    """
    order = np.lexsort(X.T[::-1])
    ordered = X[order]
    starts = np.ones(len(X), dtype=bool)
    starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    cell_of_row = np.empty(len(X), dtype=np.intp)
    cell_of_row[order] = np.cumsum(starts) - 1
    return np.ascontiguousarray(ordered[starts]), cell_of_row
