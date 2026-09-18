"""Interpretable k-nearest-neighbour classification on low-dimensional feature subspaces.

The package provides :class:`SubspaceKNNClassifier`, a scikit-learn compatible
classifier that fits one k-nearest-neighbour model per small subset of features,
ranks those subspaces by cross-validated performance, and combines the best of
them by weighted voting. Because every member of the ensemble lives in a space of
one, two or three features, each prediction can be explained by looking at the
neighbourhoods that produced it.
"""

from subspaceknn._classifier import SubspaceKNNClassifier
from subspaceknn._explanation import Explanation, SubspaceVote

__all__ = ["Explanation", "SubspaceKNNClassifier", "SubspaceVote"]
__version__ = "0.1.0"
