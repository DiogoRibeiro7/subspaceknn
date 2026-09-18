"""Ensemble of k-nearest-neighbour classifiers on low-dimensional feature subspaces."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import ceil, comb
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from subspaceknn._explanation import Explanation, SubspaceVote

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray
    from sklearn.model_selection import BaseCrossValidator

Voting = Literal["soft", "hard"]
Weighting = Literal["score", "uniform"]


class SubspaceKNNClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Weighted vote of k-nearest-neighbour classifiers fitted on small feature subspaces.

    The estimator enumerates every subset of ``subspace_size`` features (or every
    subset of each size when a sequence of sizes is given), fits a
    :class:`~sklearn.neighbors.KNeighborsClassifier` on each subset, and scores
    it by cross-validation on the training data. The ``n_subspaces`` best
    subspaces then vote on new samples, each weighted by its cross-validated
    score. Because the members of the ensemble live in spaces of one, two or
    three features, every prediction can be explained by looking at the
    neighbourhoods that produced it; see :meth:`explain`.

    The method generalises the interpretable kNN (ikNN) idea of Brett Kennedy,
    which uses pairs of features only, to subspaces of any small size. This is
    an independent implementation that shares no code with the original.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours used by every subspace model.
    subspace_size : int or sequence of int, default=2
        Number of features in each subspace. A sequence enumerates subspaces of
        every listed size. Sizes larger than the number of features are ignored.
    n_subspaces : int or None, default=5
        Number of best-scoring subspaces used for prediction. ``None`` uses every
        candidate subspace.
    max_candidates : int or None, default=100
        Soft cap on the number of candidate subspaces that are cross-validated.
        When the number of subsets exceeds it, features are first screened by the
        cross-validated score of their one-dimensional model and only the
        best-scoring features are combined, as many as keep the candidate count
        within the cap. ``None`` evaluates every subset, which grows
        combinatorially with the number of features.
    voting : {"soft", "hard"}, default="soft"
        ``"soft"`` averages the class probabilities of the subspace models,
        ``"hard"`` averages their one-hot predictions.
    weighting : {"score", "uniform"}, default="score"
        ``"score"`` weights each subspace by its cross-validated score (negative
        scores are clipped to zero), ``"uniform"`` gives every subspace the same
        weight.
    cv : int or cross-validation generator, default=5
        Cross-validation used to score subspaces. An integer selects stratified
        k-fold with that many splits, reduced automatically when a class has fewer
        samples than splits. When the training set is too small to
        cross-validate at all, the resubstitution score on the training data is
        used instead.
    scoring : str or callable, default="f1_macro"
        Any scikit-learn scorer. Score-based weighting assumes higher is better
        and scores are non-negative.
    knn_weights : {"uniform", "distance"}, default="uniform"
        Neighbour weighting passed to the subspace models.
    metric : str, default="minkowski"
        Distance metric passed to the subspace models.
    n_jobs : int or None, default=None
        Parallelism for cross-validation, passed to
        :func:`~sklearn.model_selection.cross_val_score`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names, only when ``X`` had string column names.
    screened_features_ : ndarray of shape (n_screened,)
        Indices of the features retained after screening, all features when no
        screening was necessary.
    feature_screening_scores_ : ndarray of shape (n_features_in_,) or None
        Cross-validated score of each feature's one-dimensional model, only when
        screening took place.
    candidate_subspaces_ : list of tuple of int
        Every subspace that was cross-validated, in enumeration order.
    candidate_scores_ : ndarray of shape (n_candidates,)
        Cross-validated score of each candidate subspace.
    subspaces_ : list of tuple of int
        Subspaces used for prediction, from the best to the worst score.
    subspace_scores_ : ndarray of shape (n_selected,)
        Scores of the selected subspaces.
    subspace_weights_ : ndarray of shape (n_selected,)
        Normalised voting weights of the selected subspaces; they sum to one.
    estimators_ : list of KNeighborsClassifier
        Fitted subspace models, aligned with ``subspaces_``.
    feature_scores_ : ndarray of shape (n_features_in_,)
        Mean score of the selected subspaces containing each feature, zero for
        features that appear in none. A coarse measure of feature relevance.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from subspaceknn import SubspaceKNNClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = SubspaceKNNClassifier(subspace_size=(1, 2)).fit(X, y)
    >>> clf.subspaces_[0]
    (2, 3)
    >>> clf.explain(X[:1])[0].votes[0].feature_names
    ('x2', 'x3')
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        subspace_size: int | Sequence[int] = 2,
        n_subspaces: int | None = 5,
        max_candidates: int | None = 100,
        voting: Voting = "soft",
        weighting: Weighting = "score",
        cv: int | BaseCrossValidator = 5,
        scoring: str | Callable[..., float] = "f1_macro",
        knn_weights: Literal["uniform", "distance"] = "uniform",
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.subspace_size = subspace_size
        self.n_subspaces = n_subspaces
        self.max_candidates = max_candidates
        self.voting = voting
        self.weighting = weighting
        self.cv = cv
        self.scoring = scoring
        self.knn_weights = knn_weights
        self.metric = metric
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ fitting

    def fit(self, X: ArrayLike, y: ArrayLike) -> SubspaceKNNClassifier:
        """Fit one subspace model per candidate subspace and select the best.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
            The fitted estimator.
        """
        X_arr, y_arr = validate_data(self, X, y, dtype="numeric")
        check_classification_targets(y_arr)
        self._check_hyperparameters()
        n_samples, n_features = X_arr.shape
        if self.n_neighbors > n_samples:
            raise ValueError(
                "Expected n_neighbors <= n_samples, got "
                f"n_neighbors={self.n_neighbors} and n_samples={n_samples}.",
            )

        self.classes_ = unique_labels(y_arr)
        y_encoded = np.searchsorted(self.classes_, y_arr)
        sizes = self._subspace_sizes(n_features)
        cv = self._cross_validator(y_encoded)

        features = np.arange(n_features)
        self.feature_screening_scores_: NDArray[np.float64] | None = None
        total = sum(comb(n_features, size) for size in sizes)
        if self.max_candidates is not None and total > self.max_candidates:
            singletons = [(index,) for index in range(n_features)]
            screening = self._score_subspaces(singletons, X_arr, y_encoded, cv)
            self.feature_screening_scores_ = screening
            keep = self._n_features_to_keep(sizes, n_features, self.max_candidates)
            features = np.sort(np.argsort(-screening, kind="stable")[:keep])
        self.screened_features_ = features

        feature_list = [int(index) for index in features]
        candidates = [combo for size in sizes for combo in combinations(feature_list, size)]
        scores = self._score_subspaces(candidates, X_arr, y_encoded, cv)
        self.candidate_subspaces_ = candidates
        self.candidate_scores_ = scores

        order = np.argsort(-scores, kind="stable")
        if self.n_subspaces is not None:
            order = order[: self.n_subspaces]
        self.subspaces_ = [candidates[index] for index in order]
        self.subspace_scores_ = scores[order]
        self.subspace_weights_ = self._weights(self.subspace_scores_)
        self.estimators_ = [
            self._make_knn().fit(X_arr[:, list(subspace)], y_encoded)
            for subspace in self.subspaces_
        ]
        self.feature_scores_ = self._feature_scores(n_features)
        return self

    def _check_hyperparameters(self) -> None:
        if not _is_positive_integer(self.n_neighbors):
            raise ValueError(f"n_neighbors must be a positive integer, got {self.n_neighbors!r}.")
        sizes = self._requested_sizes()
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError(
                "subspace_size must be a positive integer or a non-empty sequence of "
                f"positive integers, got {self.subspace_size!r}.",
            )
        if self.n_subspaces is not None and not _is_positive_integer(self.n_subspaces):
            raise ValueError(
                f"n_subspaces must be None or a positive integer, got {self.n_subspaces!r}.",
            )
        if self.max_candidates is not None and not _is_positive_integer(self.max_candidates):
            raise ValueError(
                f"max_candidates must be None or a positive integer, got {self.max_candidates!r}.",
            )
        if self.voting not in ("soft", "hard"):
            raise ValueError(f"voting must be 'soft' or 'hard', got {self.voting!r}.")
        if self.weighting not in ("score", "uniform"):
            raise ValueError(f"weighting must be 'score' or 'uniform', got {self.weighting!r}.")
        if _is_integer(self.cv) and self.cv < 2:  # noqa: PLR2004
            raise ValueError(f"cv must be at least 2 when given as an integer, got {self.cv!r}.")

    def _requested_sizes(self) -> list[int]:
        value: object = self.subspace_size
        if _is_integer(value):
            return [value]
        if isinstance(value, str) or not isinstance(value, Iterable):
            return []
        sizes = list(value)
        if not all(_is_integer(size) for size in sizes):
            return []
        return [int(size) for size in sizes]

    def _subspace_sizes(self, n_features: int) -> list[int]:
        requested = sorted(set(self._requested_sizes()))
        sizes = [size for size in requested if size <= n_features]
        if not sizes:
            raise ValueError(
                f"No feature subspace of size {requested} fits in n_features={n_features}; "
                "reduce subspace_size or add features.",
            )
        return sizes

    def _cross_validator(self, y_encoded: NDArray[np.intp]) -> BaseCrossValidator | None:
        """Return the splitter used to score subspaces, or None for resubstitution scoring."""
        if not _is_integer(self.cv):
            return self.cv
        n_samples = len(y_encoded)
        smallest_class = int(np.bincount(y_encoded).min())
        n_splits = min(self.cv, smallest_class)
        if n_splits < 2:  # noqa: PLR2004
            return None
        smallest_training_fold = n_samples - ceil(n_samples / n_splits)
        if smallest_training_fold < self.n_neighbors:
            return None
        return StratifiedKFold(n_splits=n_splits)

    def _make_knn(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.knn_weights,
            metric=self.metric,
        )

    def _score_subspaces(
        self,
        subspaces: Sequence[tuple[int, ...]],
        X: NDArray[np.float64],
        y_encoded: NDArray[np.intp],
        cv: BaseCrossValidator | None,
    ) -> NDArray[np.float64]:
        if cv is None:
            return self._resubstitution_scores(subspaces, X, y_encoded)
        scores = np.empty(len(subspaces), dtype=np.float64)
        for position, subspace in enumerate(subspaces):
            fold_scores = cross_val_score(
                self._make_knn(),
                X[:, list(subspace)],
                y_encoded,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                error_score="raise",
            )
            scores[position] = float(np.mean(fold_scores))
        return scores

    def _resubstitution_scores(
        self,
        subspaces: Sequence[tuple[int, ...]],
        X: NDArray[np.float64],
        y_encoded: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Score each subspace on its own training data.

        Used only when the training set is too small to cross-validate.
        """
        scorer = get_scorer(self.scoring)
        scores = np.empty(len(subspaces), dtype=np.float64)
        for position, subspace in enumerate(subspaces):
            X_sub = X[:, list(subspace)]
            model = self._make_knn().fit(X_sub, y_encoded)
            scores[position] = float(scorer(model, X_sub, y_encoded))
        return scores

    @staticmethod
    def _n_features_to_keep(sizes: Sequence[int], n_features: int, max_candidates: int) -> int:
        """Largest feature count whose subspace count stays within ``max_candidates``."""
        floor = max(sizes)
        keep = floor
        for count in range(floor, n_features + 1):
            if sum(comb(count, size) for size in sizes) <= max_candidates:
                keep = count
            else:
                break
        return keep

    def _weights(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weighting == "score":
            clipped = np.clip(scores, 0.0, None)
            if clipped.sum() > 0.0:
                return np.asarray(clipped / clipped.sum(), dtype=np.float64)
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)

    def _feature_scores(self, n_features: int) -> NDArray[np.float64]:
        totals = np.zeros(n_features, dtype=np.float64)
        counts = np.zeros(n_features, dtype=np.float64)
        for subspace, score in zip(self.subspaces_, self.subspace_scores_, strict=True):
            for feature in subspace:
                totals[feature] += score
                counts[feature] += 1.0
        return np.asarray(np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0))

    # --------------------------------------------------------------- predicting

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the weighted average of the subspace models' class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities aligned with ``classes_``; each row sums to one.
        """
        check_is_fitted(self)
        X_arr = validate_data(self, X, reset=False, dtype="numeric")
        votes = self._subspace_votes(X_arr)
        proba = np.tensordot(self.subspace_weights_, votes, axes=1)
        return self._normalise(proba)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the highest ensemble probability for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)])

    def explain(self, X: ArrayLike) -> list[Explanation]:
        """Explain the prediction for every sample in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        list of Explanation
            One explanation per sample, each listing the vote of every subspace
            used by the ensemble, ordered from the highest to the lowest weight.
        """
        check_is_fitted(self)
        X_arr = validate_data(self, X, reset=False, dtype="numeric")
        votes = self._subspace_votes(X_arr)
        proba = self._normalise(np.tensordot(self.subspace_weights_, votes, axes=1))
        names = self._feature_names()
        explanations: list[Explanation] = []
        for row in range(X_arr.shape[0]):
            row_votes = tuple(
                SubspaceVote(
                    features=subspace,
                    feature_names=tuple(names[index] for index in subspace),
                    score=float(score),
                    weight=float(weight),
                    probabilities=votes[position, row],
                    prediction=self.classes_[int(np.argmax(votes[position, row]))],
                )
                for position, (subspace, score, weight) in enumerate(
                    zip(
                        self.subspaces_, self.subspace_scores_, self.subspace_weights_, strict=True
                    ),
                )
            )
            explanations.append(
                Explanation(
                    prediction=self.classes_[int(np.argmax(proba[row]))],
                    probabilities=proba[row],
                    classes=self.classes_,
                    votes=row_votes,
                ),
            )
        return explanations

    def _subspace_votes(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return an array of shape (n_selected, n_samples, n_classes) with each subspace's vote."""
        n_classes = len(self.classes_)
        votes = np.empty((len(self.estimators_), X.shape[0], n_classes), dtype=np.float64)
        for position, (estimator, subspace) in enumerate(
            zip(self.estimators_, self.subspaces_, strict=True),
        ):
            X_sub = X[:, list(subspace)]
            if self.voting == "hard":
                votes[position] = np.eye(n_classes)[estimator.predict(X_sub)]
            else:
                votes[position] = estimator.predict_proba(X_sub)
        return votes

    def _normalise(self, proba: NDArray[np.float64]) -> NDArray[np.float64]:
        row_sums = proba.sum(axis=1, keepdims=True)
        uniform = np.full_like(proba, 1.0 / proba.shape[1])
        return np.asarray(np.divide(proba, row_sums, out=uniform, where=row_sums > 0))

    def _feature_names(self) -> list[str]:
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            return [f"x{index}" for index in range(self.n_features_in_)]
        return [str(name) for name in names]


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_positive_integer(value: object) -> TypeGuard[int]:
    return _is_integer(value) and value > 0
