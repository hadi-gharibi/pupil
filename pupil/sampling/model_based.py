from __future__ import annotations

from typing import Literal

import numpy as np
from pupil.types import NDArray2D
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class RankTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        return None

    def fit(self, X, y=None):
        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")
        self._bins = X.copy()
        self._bins.T.sort()
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def transform(self, X):
        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")

        return self._transform(X)

    def _transform(self, X):
        if X is None:
            X = self._bins.copy()
        X = X.copy()
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._transform_col(
                X[:, feature_idx], self._bins[:, feature_idx]
            )
        return X / self._bins.shape[0]

    def _transform_col(self, X, y):
        return np.digitize(X, y)


class LinearInterpolationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        return None

    def fit(self, X, y=None):
        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")
        self._bins = X.copy()
        self._bins.T.sort()
        # self._ranks = np.digitize(self._bins, self._bins)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def transform(self, X):

        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")
        return self._transform(X)

    def _transform(self, X):

        if X is None:
            X = self._bins.copy()
        X = X.copy()
        for feature_idx in range(X.shape[1]):

            X[:, feature_idx] = self._transform_col(
                X[:, feature_idx], self._bins[:, feature_idx]
            )
        return X / self._bins.shape[0]

    def _transform_col(self, X, y):
        _ranks = np.digitize(y, np.flip(y))
        return np.interp(X, y, list(range(len(y))))


class ModelBasedSampler:
    def __init__(self, ranker):
        self.ranker = ranker
        self.indices_ = np.array([])

    @classmethod
    def from_strategy(
        cls, strategy: Literal["rank", "quantile", "linear"] = "linear"
    ) -> ModelBasedSampler:
        if strategy not in ["rank", "quantile", "linear"]:
            raise ValueError("strategy must be one of  ['rank','quantile']")
        if strategy == "rank":
            ranker = RankTransformer()
        elif strategy == "quantile":
            ranker = QuantileTransformer()
        elif strategy == "linear":
            ranker = LinearInterpolationTransformer()
        return cls(ranker=ranker)

    def fit(self, X: NDArray2D):
        if hasattr(self.ranker, "n_quantiles"):
            self.ranker.n_quantiles = len(X)
        self.ranker.fit(X)

    def predict(self, X: NDArray2D):

        X = 1 - self.ranker.transform(X)
        X = X.mean(axis=1)
        temp = X.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(X))
        self.indices_ = ranks
        return ranks


if __name__ == "__main__":
    train = np.array(
        [
            [2.43, 2.23, 1.74, 0.89, 0.44, 0.23, -0.34, -0.36, -0.42, 1.12],
            [2.43, 2.23, 1.74, 0.89, 0.44, 0.23, -0.34, -0.36, -0.42, 1.12],
        ]
    ).T

    test = np.array([[-0.35, -0.35], [1, 1]])

    sampler = ModelBasedSampler.from_strategy("quantile")
    sampler.fit(train)
    print(sampler.predict(test))

    sampler = ModelBasedSampler.from_strategy("rank")
    sampler.fit(train)
    print(sampler.predict(test))

    sampler = ModelBasedSampler.from_strategy("linear")
    sampler.fit(train)
    print(sampler.predict(test))
