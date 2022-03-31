import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RankTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")
        self._X = X
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def transform(self, X):
        if len(X.shape) != 2:
            raise ValueError("X need to be 2D")
        return self._transform(X)

    def _transform(self, X):
        if X is None:
            X = self._X.copy()
        print(X.shape)
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._transform(X[:, feature_idx])
        return X

    def _transform_col(self, X):
        return np.digitize(X, self._X)
