from __future__ import annotations

import math
from typing import Callable

import numpy as np
from pupil.types import NDArray2D


def least_confidence(prob_dist: NDArray2D) -> np.ndarray:

    """
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is most uncertain.

    Example:

    Assumes probability distribution is a numpy array, like ``np.array ([[0.0321, 0.6439, 0.0871, 0.2369]])``
    The restults will be ``(1 – 0.6439) × (4 / 3) = 0.4748``

    Args:
        prob_dist (NDArray2D): a 2D numpy array of real numbers between 0 and 1
        each row is a data point, and each column shows the probability of that class

    Returns:
        np.ndarray: shape(n_rows)

    """
    simple_least_conf = np.max(prob_dist, axis=1)
    num_labels = prob_dist.shape[1]
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf


def margin_confidence(prob_dist: NDArray2D) -> np.ndarray:
    """
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is most uncertain.

    Example:

    Assumes probability distribution is a numpy array, like:
    ``np.array([[0.0321, 0.6439, 0.0871, 0.2369]])``
    The results would will be ``1.0 - (0.6439 - 0.2369) = 0.5930``

    Args:
        prob_dist (NDArray2D): a 2D numpy array of real numbers between 0 and 1
        each row is a data point, and each column shows the probability of that class.

    Returns:
        np.ndarray: shape(n_rows)

    """
    prob_dist = np.sort(prob_dist)
    difference = prob_dist[:, -1] - prob_dist[:, -2]  # type: ignore
    margin_conf = 1 - difference
    return margin_conf


def ratio_confidence(prob_dist: NDArray2D) -> np.ndarray:
    """
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is most uncertain.
    Example:

    Assumes probability distribution is a numpy array, like
    ``np.array ***([[0.0321, 0.6439, 0.0871, 0.2369]])``
    The results will be
    ``0.6439 / 0.2369 = 2.71828``

    Args:
        prob_dist (NDArray2D): a 2D numpy array of real numbers between 0 and 1
        each row is a data point, and each column shows the probability of that class

    Returns:
        np.ndarray: shape(n_rows)

    """
    prob_dist = np.sort(prob_dist)
    difference = prob_dist[:, -1] / prob_dist[:, -2]
    return difference


def entropy_based(prob_dist: NDArray2D) -> np.ndarray:
    """
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is most uncertain.

    Example:

    Assumes probability distribution is a numpy array, like:
    ``np.array([[0.0321, 0.6439, 0.0871, 0.2369]])``
    The results will be

    ``P(y|x) log2(P(y|x)) = 0 – SUM(–0.159, –0.409, –0.307, –0.492) = 1.367``

    ``1.367 / log2(n_classes = 4) = 0.684``

    Args:
        prob_dist (NDArray2D): a 2D numpy array of real numbers between 0 and 1
        each row is a data point, and each column shows the probability of that class

    Returns:
        np.ndarray: shape(n_rows)
    """
    log_probs = prob_dist * np.log2(prob_dist)
    raw_entropy = 0 - np.sum(log_probs, axis=1)
    normalized_entropy = raw_entropy / math.log2(prob_dist.shape[1])
    return normalized_entropy


class UncertaintySampler:
    """
    Uncertainty sampling is a set of techniques for identifying
    unlabeled items that are near a decision boundary in your
    current machine learning model.
    """

    def __init__(self, sampling_strategy: Callable[[np.ndarray], np.ndarray]):
        self.sampling_strategy = sampling_strategy
        self.indices_ = None

    @classmethod
    def from_strategy(cls, strategy: str) -> UncertaintySampler:
        """classmethod to help picking the sampling strategy

        Args:
            strategy (str): Should be one of:
                ``['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based']``

        Raises:
            ValueError: If strategy is not in the valid list

        Returns:
            UncertaintySampler:
        """
        strategies = [
            "least_confidence",
            "margin_confidence",
            "ratio_confidence",
            "entropy_based",
        ]
        if strategy not in strategies:
            raise ValueError(f"{strategy} must be one of {strategies}")
        return cls(eval(strategy))

    def __call__(self, prob_dist: np.ndarray) -> UncertaintySampler:
        self.fit(prob_dist)
        return self.indices_  # type: ignore

    def fit(self, prob_dist: NDArray2D) -> None:
        """Get the 2D numpy array of model predictions and retun an array on indecies
        with the order of highest to lowst uncertainty.

        Args:
            prob_dist (NDArray2D):
        """

        ranks = self.sampling_strategy(prob_dist)
        self.indices_ = np.flip(ranks.argsort())
