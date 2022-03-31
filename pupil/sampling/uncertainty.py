from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Callable, List, Optional, Union

import numpy as np
from nptyping import NDArray
from pupil.models.clustering import Clustering
from pupil.types import SeqOfIndices


def least_confidence(prob_dist: np.ndarray) -> np.ndarray:
    """
    Returns the uncertainty score of an array using
    least confidence sampling in a 0-1 range where 1 is most uncertain
    Assumes probability distribution is a pytorch tensor, like:
    tensor([0.0321, 0.6439, 0.0871, 0.2369])
    Keyword arguments:
    prob_dist -- a numpy array of real numbers between 0 and 1 that
    """
    simple_least_conf = np.max(prob_dist, axis=1)
    num_labels = prob_dist.shape[1]
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf


def margin_confidence(prob_dist: np.ndarray) -> np.ndarray:
    """
    Returns the uncertainty score of a probability distribution using
    margin of confidence sampling in 0-1 range where 1 is most uncertain
    Assumes probability distribution is a pytorch tensor, like:
    tensor([0.0321, 0.6439, 0.0871, 0.2369])
    Keyword arguments:
    prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
    """
    prob_dist = np.sort(prob_dist)
    difference = prob_dist[:, -1] - prob_dist[:, -2]  # type: ignore
    margin_conf = 1 - difference
    return margin_conf


def ratio_confidence(prob_dist: np.ndarray) -> np.ndarray:
    """
    Returns the uncertainty score of a probability distribution using
    ratio of confidence sampling in 0-1 range where 1 is most uncertain
    Assumes probability distribution is a pytorch tensor, like:
    tensor([0.0321, 0.6439, 0.0871, 0.2369])
    Keyword arguments:
    prob_dist --  a numpy array of real numbers between 0 and 1 that total to 1.0
    """
    prob_dist = np.sort(prob_dist)
    difference = prob_dist[:, -1] / prob_dist[:, -2]
    return difference


def entropy_based(prob_dist: np.ndarray) -> np.ndarray:
    """
    Returns uncertainty score of a probability distribution using entropy
    Assumes probability distribution is a pytorch tensor, like:
    tensor([0.0321, 0.6439, 0.0871, 0.2369])
    Keyword arguments:
    prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
    """
    log_probs = prob_dist * np.log2(prob_dist)
    raw_entropy = 0 - np.sum(log_probs, axis=1)
    normalized_entropy = raw_entropy / math.log2(prob_dist.shape[1])
    return normalized_entropy


class UncertaintySampler:
    def __init__(self, sampling_strategy: Callable[[np.ndarray], np.ndarray]):
        self.sampling_strategy = sampling_strategy

    @classmethod
    def from_strategy(cls, strategy: str) -> UncertaintySampler:
        """classmethod to help picking the sampling strategy

        Args:
            strategy (str): Should be one of:
                {'least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based'}

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

    def __call__(self, prob_dist: np.ndarray) -> np.ndarray:
        ranks = self.sampling_strategy(prob_dist)
        return np.flip(ranks.argsort())
