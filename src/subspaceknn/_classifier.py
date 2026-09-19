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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from subspaceknn._explanation import Explanation, SubspaceVote
from subspaceknn._neighbours import TieSharingKNeighborsClassifier

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray
    from sklearn.model_selection import BaseCrossValidator

Selection = Literal["complementary", "ranked"]
Voting = Literal["soft", "hard"]
Weighting = Literal["score", "uniform"]

# Losses closer than this are treated as equal: a greedy step takes the first
# enumerated candidate among the equal best, and a vote only counts as an
# improvement when it lowers the loss by more. Losses are sums over samples whose
# last bits depend on the summation order, which differs between platforms and
# row orders; without the tolerance that noise could decide a choice.
_LOSS_TOLERANCE = 1e-12


class SubspaceKNNClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Weighted vote of k-nearest-neighbour classifiers fitted on small feature subspaces.

    The estimator fits a k-nearest-neighbour model on every subset of
    ``subspace_size`` features (or every subset of each size when a sequence of
    sizes is given) and computes each subset's out-of-fold class probabilities on
    the training data, by exact leave-one-out by default. It then chooses at most
    ``n_subspaces`` of them to vote on new samples.

    The default, complementary selection, builds the ensemble greedily: at every
    step it adds the subspace that most reduces the class-balanced Brier score of
    the ensemble's out-of-fold probabilities, allowing a subspace to be added
    again, and keeps the best ensemble found. A subspace therefore earns its place
    by what it adds to the others, not by how well it does alone, and its weight is
    the number of times it was chosen. Ranked selection instead keeps the
    subspaces with the best individual scores, as ikNN does.

    Because the members of the ensemble live in spaces of one, two or three
    features, every prediction can be explained by looking at the neighbourhoods
    that produced it; see [`explain`][subspaceknn.SubspaceKNNClassifier.explain].

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours used by every subspace model.
    subspace_size : int or sequence of int, default=2
        Number of features in each subspace. A sequence enumerates subspaces of
        every listed size. Sizes larger than the number of features are ignored.
    n_subspaces : int or None, default=5
        Maximum number of distinct subspaces used for prediction, that is, the
        number of pictures an explanation contains. Complementary selection may
        use fewer when more would not improve the ensemble. ``None`` removes the
        limit.
    selection : {"complementary", "ranked"}, default="complementary"
        ``"complementary"`` builds the ensemble by greedy forward selection on
        out-of-fold probabilities, as described above. ``"ranked"`` keeps the
        ``n_subspaces`` subspaces with the highest individual ``scoring`` and
        weights them according to ``weighting``.
    max_votes : int, default=50
        Number of greedy steps of complementary selection. Each step casts one
        vote and the best ensemble over all steps is kept, so the weights are
        multiples of ``1 / n_votes`` for some ``n_votes <= max_votes``. Ignored by
        ranked selection.
    balance_classes : bool, default=True
        Whether the Brier score minimised by complementary selection weights each
        sample inversely to its class frequency, so that every class counts
        equally. ``False`` weights samples equally. Ignored by ranked selection.
    max_candidates : int or None, default=1000
        Soft cap on the number of candidate subspaces. When the number of subsets
        exceeds it, features are first screened by the ``scoring`` of their
        one-dimensional model and only the best-scoring features are combined, as
        many as keep the candidate count within the cap. ``None`` evaluates every
        subset, which grows combinatorially with the number of features.
    voting : {"soft", "hard"}, default="soft"
        ``"soft"`` averages the class probabilities of the subspace models,
        ``"hard"`` averages their one-hot predictions. Complementary selection
        optimises whichever of the two is used for prediction.
    weighting : {"score", "uniform"}, default="score"
        Weights of ranked selection: ``"score"`` weights each subspace by its
        individual score (negative scores are clipped to zero), ``"uniform"``
        gives every subspace the same weight. Ignored by complementary selection,
        whose weights are its vote counts.
    cv : "loo", int or cross-validation generator, default="loo"
        How the out-of-fold probabilities are computed. ``"loo"`` is exact
        leave-one-out, obtained from a single neighbour query per subspace. An
        integer selects stratified k-fold with that many splits, reduced
        automatically when a class has fewer samples than splits; a splitter must
        partition the samples. When the training set is too small for the chosen
        scheme, the probabilities are computed on the training data itself.
    scoring : str or callable, default="f1_macro"
        Any scikit-learn scorer, evaluated on each subspace's out-of-fold
        predictions. It screens features, ranks subspaces under ranked selection,
        and is reported as each subspace's score. Higher must be better.
    knn_weights : {"uniform", "distance"}, default="uniform"
        Neighbour weighting passed to the subspace models.
    metric : str, default="minkowski"
        Distance metric passed to the subspace models.
    n_jobs : int or None, default=None
        Parallelism of the neighbour queries, or of the cross-validation when
        ``cv`` is not ``"loo"``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    n_features_in_ : int
        Number of features seen during [`fit`][subspaceknn.SubspaceKNNClassifier.fit].
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names, only when ``X`` had string column names.
    screened_features_ : ndarray of shape (n_screened,)
        Indices of the features retained after screening, all features when no
        screening was necessary.
    feature_screening_scores_ : ndarray of shape (n_features_in_,) or None
        Out-of-fold score of each feature's one-dimensional model, only when
        screening took place.
    candidate_subspaces_ : list of tuple of int
        Every subspace that was evaluated, in enumeration order.
    candidate_scores_ : ndarray of shape (n_candidates,)
        Out-of-fold score of each candidate subspace on its own.
    subspaces_ : list of tuple of int
        Subspaces used for prediction, from the highest to the lowest weight.
    subspace_scores_ : ndarray of shape (n_selected,)
        Individual scores of the selected subspaces.
    subspace_weights_ : ndarray of shape (n_selected,)
        Normalised voting weights of the selected subspaces; they sum to one.
    selection_path_ : list of tuple or None
        Under complementary selection, one ``(subspace, loss)`` pair per vote of
        the final ensemble, in the order the votes were added, where ``loss`` is
        the ensemble's out-of-fold Brier score after that vote, class-balanced
        when ``balance_classes`` is true. ``None`` under ranked selection.
    estimators_ : list of TieSharingKNeighborsClassifier
        Fitted subspace models, aligned with ``subspaces_``. Points tied at the
        distance of the k-th neighbour share the remaining votes, so predictions
        do not depend on the order of the training data or on the platform.
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
        selection: Selection = "complementary",
        max_votes: int = 50,
        balance_classes: bool = True,
        max_candidates: int | None = 1000,
        voting: Voting = "soft",
        weighting: Weighting = "score",
        cv: Literal["loo"] | int | BaseCrossValidator = "loo",
        scoring: str | Callable[..., float] = "f1_macro",
        knn_weights: Literal["uniform", "distance"] = "uniform",
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.subspace_size = subspace_size
        self.n_subspaces = n_subspaces
        self.selection = selection
        self.max_votes = max_votes
        self.balance_classes = balance_classes
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
        """Evaluate every candidate subspace and select the ensemble.

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
        splitter = self._splitter(y_encoded)

        features = np.arange(n_features)
        self.feature_screening_scores_: NDArray[np.float64] | None = None
        total = sum(comb(n_features, size) for size in sizes)
        if self.max_candidates is not None and total > self.max_candidates:
            singletons = [(index,) for index in range(n_features)]
            screening, _ = self._evaluate(singletons, X_arr, y_encoded, splitter, keep=False)
            self.feature_screening_scores_ = screening
            keep = self._n_features_to_keep(sizes, n_features, self.max_candidates)
            features = np.sort(np.argsort(-screening, kind="stable")[:keep])
        self.screened_features_ = features

        feature_list = [int(index) for index in features]
        candidates = [combo for size in sizes for combo in combinations(feature_list, size)]
        scores, votes = self._evaluate(
            candidates, X_arr, y_encoded, splitter, keep=self.selection == "complementary"
        )
        self.candidate_subspaces_ = candidates
        self.candidate_scores_ = scores

        if votes is not None:
            order, weights, path = _complementary_selection(
                votes,
                y_encoded,
                self._sample_weight(y_encoded),
                budget=self.n_subspaces,
                max_votes=self.max_votes,
            )
            self.selection_path_: list[tuple[tuple[int, ...], float]] | None = [
                (candidates[index], loss) for index, loss in path
            ]
        else:
            order = np.argsort(-scores, kind="stable")
            if self.n_subspaces is not None:
                order = order[: self.n_subspaces]
            weights = self._ranked_weights(scores[order])
            self.selection_path_ = None
        self.subspaces_ = [candidates[index] for index in order]
        self.subspace_scores_ = scores[order]
        self.subspace_weights_ = weights
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
        self._check_selection_hyperparameters()
        if self.voting not in ("soft", "hard"):
            raise ValueError(f"voting must be 'soft' or 'hard', got {self.voting!r}.")
        if (isinstance(self.cv, str) and self.cv != "loo") or (
            _is_integer(self.cv) and self.cv < 2  # noqa: PLR2004
        ):
            raise ValueError(
                "cv must be 'loo', an integer of at least 2 or a cross-validation "
                f"splitter, got {self.cv!r}.",
            )

    def _check_selection_hyperparameters(self) -> None:
        if self.selection not in ("complementary", "ranked"):
            raise ValueError(
                f"selection must be 'complementary' or 'ranked', got {self.selection!r}.",
            )
        if not _is_positive_integer(self.max_votes):
            raise ValueError(f"max_votes must be a positive integer, got {self.max_votes!r}.")
        if self.balance_classes not in (True, False):
            raise ValueError(
                f"balance_classes must be True or False, got {self.balance_classes!r}.",
            )
        if self.weighting not in ("score", "uniform"):
            raise ValueError(f"weighting must be 'score' or 'uniform', got {self.weighting!r}.")

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

    def _splitter(self, y_encoded: NDArray[np.intp]) -> str | BaseCrossValidator | None:
        """Return "loo", a splitter, or None when only resubstitution is possible."""
        n_samples = len(y_encoded)
        if isinstance(self.cv, str):
            return "loo" if n_samples - 1 >= self.n_neighbors else None
        if not _is_integer(self.cv):
            return self.cv
        smallest_class = int(np.bincount(y_encoded).min())
        n_splits = min(self.cv, smallest_class)
        if n_splits < 2:  # noqa: PLR2004
            return None
        smallest_training_fold = n_samples - ceil(n_samples / n_splits)
        if smallest_training_fold < self.n_neighbors:
            return None
        return StratifiedKFold(n_splits=n_splits)

    def _make_knn(self) -> TieSharingKNeighborsClassifier:
        return TieSharingKNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.knn_weights,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def _evaluate(
        self,
        subspaces: Sequence[tuple[int, ...]],
        X: NDArray[np.float64],
        y_encoded: NDArray[np.intp],
        splitter: str | BaseCrossValidator | None,
        *,
        keep: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Score every subspace on its out-of-fold predictions.

        Returns the scores and, when ``keep`` is true, the out-of-fold votes as an
        array of shape (n_subspaces, n_samples, n_classes): probabilities under
        soft voting, one-hot predictions under hard voting.
        """
        n_classes = len(self.classes_)
        scorer = get_scorer(self.scoring)
        rows = np.arange(len(y_encoded)).reshape(-1, 1)
        scores = np.empty(len(subspaces), dtype=np.float64)
        votes = (
            np.empty((len(subspaces), len(y_encoded), n_classes), dtype=np.float64)
            if keep
            else None
        )
        for position, subspace in enumerate(subspaces):
            proba = self._out_of_fold_proba(X[:, list(subspace)], y_encoded, splitter)
            scores[position] = float(
                scorer(_FixedPredictions(proba), rows, y_encoded),
            )
            if votes is not None:
                votes[position] = (
                    np.eye(n_classes)[np.argmax(proba, axis=1)] if self.voting == "hard" else proba
                )
        return scores, votes

    def _out_of_fold_proba(
        self,
        X_sub: NDArray[np.float64],
        y_encoded: NDArray[np.intp],
        splitter: str | BaseCrossValidator | None,
    ) -> NDArray[np.float64]:
        if splitter is None:
            return self._make_knn().fit(X_sub, y_encoded).predict_proba(X_sub)
        if isinstance(splitter, str):
            return self._make_knn().fit(X_sub, y_encoded).leave_one_out_proba()
        return np.asarray(
            cross_val_predict(
                self._make_knn(),
                X_sub,
                y_encoded,
                cv=splitter,
                method="predict_proba",
                n_jobs=self.n_jobs,
            ),
            dtype=np.float64,
        )

    def _sample_weight(self, y_encoded: NDArray[np.intp]) -> NDArray[np.float64]:
        if not self.balance_classes:
            return np.ones(len(y_encoded), dtype=np.float64)
        counts = np.bincount(y_encoded).astype(np.float64)
        return np.asarray(len(y_encoded) / (len(counts) * counts[y_encoded]), dtype=np.float64)

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

    def _ranked_weights(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
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

    @staticmethod
    def _normalise(proba: NDArray[np.float64]) -> NDArray[np.float64]:
        row_sums = proba.sum(axis=1, keepdims=True)
        uniform = np.full_like(proba, 1.0 / proba.shape[1])
        return np.asarray(np.divide(proba, row_sums, out=uniform, where=row_sums > 0))

    def _feature_names(self) -> list[str]:
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            return [f"x{index}" for index in range(self.n_features_in_)]
        return [str(name) for name in names]


def _complementary_selection(
    votes: NDArray[np.float64],
    y_encoded: NDArray[np.intp],
    sample_weight: NDArray[np.float64],
    *,
    budget: int | None,
    max_votes: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64], list[tuple[int, float]]]:
    """Greedy forward selection, with replacement, of the votes that minimise the Brier score.

    Step ``t`` adds the candidate ``c`` that minimises the weighted Brier score of
    ``(T + V_c) / t``, where ``T`` is the sum of the votes chosen so far. Once
    ``budget`` distinct candidates have been chosen, only those can be added
    again. The best ensemble over all steps is returned. Losses within
    ``_LOSS_TOLERANCE`` of each other count as ties, and ties go to the candidate
    enumerated first.

    Writing ``A = T / t - Y``, the loss of candidate ``c`` is, up to division by
    the total sample weight,
    ``sum_i s_i |A_i|^2 + (2 / t) <V_c, s A> + (1 / t^2) sum_i s_i |V_c,i|^2``,
    so each step costs one matrix-vector product over the candidates and never
    materialises the candidate ensembles.

    Parameters
    ----------
    votes : ndarray of shape (n_candidates, n_samples, n_classes)
        Out-of-fold probabilities (or one-hot predictions) of every candidate.
    y_encoded : ndarray of shape (n_samples,)
        Encoded class labels.
    sample_weight : ndarray of shape (n_samples,)
        Weights of the samples in the Brier score.
    budget : int or None
        Maximum number of distinct candidates.
    max_votes : int
        Number of greedy steps.

    Returns
    -------
    order : ndarray of int
        Chosen candidates, by decreasing vote count, ties by first selection.
    weights : ndarray of float
        Vote count of each chosen candidate divided by the number of votes.
    path : list of (int, float)
        Candidate added and ensemble loss after each vote of the returned ensemble.
    """
    n_candidates, n_samples, n_classes = votes.shape
    onehot = np.eye(n_classes)[y_encoded]
    total_weight = float(sample_weight.sum())
    flat = votes.reshape(n_candidates, n_samples * n_classes)
    squares = np.einsum("kic,kic,i->k", votes, votes, sample_weight)

    counts = np.zeros(n_candidates, dtype=np.intp)
    running = np.zeros((n_samples, n_classes), dtype=np.float64)
    steps: list[tuple[int, float]] = []
    best_loss = np.inf
    best_step = 0
    for step in range(1, max_votes + 1):
        residual = running / step - onehot
        weighted = sample_weight[:, None] * residual
        base = float(np.sum(weighted * residual))
        losses = (base + 2.0 * (flat @ weighted.ravel()) / step + squares / step**2) / total_weight
        if budget is not None and np.count_nonzero(counts) >= budget:
            losses[counts == 0] = np.inf
        # The first enumerated candidate among those within the tolerance of the best.
        pick = int(np.flatnonzero(losses <= losses.min() + _LOSS_TOLERANCE)[0])
        counts[pick] += 1
        running += votes[pick]
        loss = float(losses[pick])
        steps.append((pick, loss))
        if loss < best_loss - _LOSS_TOLERANCE:
            best_loss = loss
            best_step = step

    path = steps[:best_step]
    final = np.zeros(n_candidates, dtype=np.intp)
    first_step: dict[int, int] = {}
    for position, (candidate, _) in enumerate(path):
        final[candidate] += 1
        first_step.setdefault(candidate, position)
    order = np.array(
        sorted(first_step, key=lambda candidate: (-final[candidate], first_step[candidate])),
        dtype=np.intp,
    )
    weights = final[order] / float(best_step)
    return order, np.asarray(weights, dtype=np.float64), path


class _FixedPredictions(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Serve precomputed out-of-fold probabilities to a scikit-learn scorer.

    The scorer passes row indices in place of ``X``, so any scorer, whether it
    needs predictions or probabilities, is evaluated on the out-of-fold values.
    """

    def __init__(self, probabilities: NDArray[np.float64]) -> None:
        self.probabilities = probabilities
        self.classes_ = np.arange(probabilities.shape[1])

    def predict_proba(self, X: NDArray[np.intp]) -> NDArray[np.float64]:
        return self.probabilities[np.asarray(X, dtype=np.intp).ravel()]

    def predict(self, X: NDArray[np.intp]) -> NDArray[np.intp]:
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_positive_integer(value: object) -> TypeGuard[int]:
    return _is_integer(value) and value > 0
