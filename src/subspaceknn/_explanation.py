"""Structured explanations of individual predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SubspaceVote:
    """The contribution of one feature subspace to a single prediction.

    Attributes
    ----------
    features : tuple of int
        Column indices of the features that span the subspace.
    feature_names : tuple of str
        Names of those features, taken from ``feature_names_in_`` when the
        estimator was fitted on a data frame and ``x<i>`` otherwise.
    score : float
        Cross-validated score of the subspace on the training data.
    weight : float
        Normalised weight of the subspace in the ensemble vote. Weights sum to one
        across the subspaces used for prediction.
    probabilities : ndarray of shape (n_classes,)
        Class probabilities produced by this subspace alone, aligned with
        ``Explanation.classes``. Under hard voting this is a one-hot vector.
    prediction : Any
        Class label predicted by this subspace alone.
    """

    features: tuple[int, ...]
    feature_names: tuple[str, ...]
    score: float
    weight: float
    probabilities: NDArray[np.float64]
    prediction: Any


@dataclass(frozen=True)
class Explanation:
    """How the ensemble arrived at the prediction for one sample.

    Attributes
    ----------
    prediction : Any
        Class label predicted by the ensemble.
    probabilities : ndarray of shape (n_classes,)
        Ensemble class probabilities, the weighted average of the votes.
    classes : ndarray of shape (n_classes,)
        Class labels, in the order used by ``probabilities``.
    votes : tuple of SubspaceVote
        One entry per subspace used for prediction, ordered from the highest to
        the lowest weight.
    """

    prediction: Any
    probabilities: NDArray[np.float64]
    classes: NDArray[Any]
    votes: tuple[SubspaceVote, ...]

    def agreement(self) -> float:
        """Return the weighted fraction of subspaces that voted for the ensemble prediction."""
        return float(
            sum(vote.weight for vote in self.votes if vote.prediction == self.prediction),
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return the votes as plain dictionaries, one per subspace.

        The result is suitable for ``pandas.DataFrame.from_records`` and contains,
        for every vote, the feature names, the subspace score and weight, the
        subspace prediction, whether it agrees with the ensemble, and one
        ``p(<class>)`` column per class.
        """
        records: list[dict[str, Any]] = []
        for vote in self.votes:
            record: dict[str, Any] = {
                "features": ", ".join(vote.feature_names),
                "score": vote.score,
                "weight": vote.weight,
                "prediction": vote.prediction,
                "agrees": bool(vote.prediction == self.prediction),
            }
            for label, probability in zip(self.classes, vote.probabilities, strict=True):
                record[f"p({label})"] = float(probability)
            records.append(record)
        return records
