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
    still `n_neighbors`. With `weights="distance"` each vote is further
    divided by the point's distance, and points at distance zero, if there are
    any, take all the weight.

    For the Euclidean, Manhattan and Chebyshev metrics, distances are recomputed
    from the coordinates with operations that round identically on every
    platform, and ties are equalities of those distances; the neighbour search only
    proposes candidates. Other metrics use the search's own distances.

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
        """Store the training data and build the neighbour search.

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
        self.classes_, self._labels = np.unique(np.asarray(y), return_inverse=True)
        self._X = X_arr
        self._search = NearestNeighbors(metric=self.metric, n_jobs=self.n_jobs).fit(X_arr)
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
        return self._votes(np.ascontiguousarray(X, dtype=np.float64), None)

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

        This equals refitting without each sample and predicting it.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Leave-one-out vote shares aligned with `classes_`.
        """
        return self._votes(self._X, np.arange(len(self._X)))

    def _votes(
        self,
        queries: NDArray[np.float64],
        own_rows: NDArray[np.intp] | None,
    ) -> NDArray[np.float64]:
        """Vote shares of `queries`; `own_rows` leaves each query's own row out."""
        n_train = len(self._X)
        k = self.n_neighbors
        available = n_train - 1 if own_rows is not None else n_train
        if k > available:
            raise ValueError(
                f"Expected n_neighbors <= {available} neighbours to choose from, got {k}.",
            )
        proba = np.empty((len(queries), len(self.classes_)), dtype=np.float64)
        pending = np.arange(len(queries))
        count = min(available, k + 1)
        while pending.size:
            rows = queries[pending]
            search_distance, index = self._candidates(
                rows, None if own_rows is None else own_rows[pending], count
            )
            reduced, distance = self._exact_distances(rows, index, search_distance)
            order = np.lexsort((index, reduced), axis=-1)
            index = np.take_along_axis(index, order, axis=1)
            reduced = np.take_along_axis(reduced, order, axis=1)
            distance = np.take_along_axis(distance, order, axis=1)
            kth = distance[:, k - 1]
            farthest = search_distance[:, -1]
            complete = (count >= available) | (farthest > kth * (1.0 + _COMPLETENESS_MARGIN))
            proba[pending[complete]] = self._shares(
                index[complete], reduced[complete], distance[complete]
            )
            pending = pending[~complete]
            count = min(available, 2 * count)
        return proba

    def _candidates(
        self,
        rows: NDArray[np.float64],
        own_rows: NDArray[np.intp] | None,
        count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Return the `count` nearest training points of each row, sorted by search distance."""
        if own_rows is None:
            distance, index = self._search.kneighbors(rows, n_neighbors=count)
            return distance, index
        distance, index = self._search.kneighbors(rows, n_neighbors=min(count + 1, len(self._X)))
        own = index == own_rows[:, None]
        # With more duplicates of a row than neighbours requested, the row itself may
        # not be among them; drop the farthest instead, as scikit-learn does.
        own[~own.any(axis=1), -1] = True
        keep = ~own
        shape = (len(rows), keep.shape[1] - 1)
        return distance[keep].reshape(shape), index[keep].reshape(shape)

    def _exact_distances(
        self,
        rows: NDArray[np.float64],
        index: NDArray[np.intp],
        search_distance: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a monotone reduced distance, used for ties, and the distance itself."""
        if self.metric not in _SQUARED_METRICS | _ABSOLUTE_METRICS | _MAXIMUM_METRICS:
            return search_distance, search_distance
        differences = [self._X[index, j] - rows[:, j : j + 1] for j in range(rows.shape[1])]
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
        index: NDArray[np.intp],
        reduced: NDArray[np.float64],
        distance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply the tie rule to candidates sorted by distance and return vote shares."""
        k = self.n_neighbors
        boundary = reduced[:, k - 1 : k]
        closer = reduced < boundary
        tied = reduced == boundary
        share = (k - closer.sum(axis=1, keepdims=True)) / tied.sum(axis=1, keepdims=True)
        weight = np.where(closer, 1.0, np.where(tied, share, 0.0))
        if self.weights == "distance":
            at_zero = reduced == 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                weighted = weight / distance
            weight = np.where(at_zero.any(axis=1, keepdims=True), at_zero * 1.0, weighted)
            weight = np.where(weight > 0.0, weight, 0.0)
        labels = self._labels[index]
        totals = np.empty((len(index), len(self.classes_)), dtype=np.float64)
        for position in range(len(self.classes_)):
            # Summing each class's weights in sorted order makes the total independent
            # of the order in which equidistant points were listed.
            own = np.where(labels == position, weight, 0.0)
            totals[:, position] = np.sort(own, axis=1).sum(axis=1)
        return np.asarray(totals / totals.sum(axis=1, keepdims=True), dtype=np.float64)
