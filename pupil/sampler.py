from typing import Protocol, Union, Optional, Any
import numpy as np
from pupil.models.clustering import Clustering

class Sampler(Protocol):
    def fit(self, data:np.ndarray, model:Optional[Clustering], labels = Optional[Any]):
        ...

    def sample(self):
        ...

class RandomArraySampler:
    def __init__(self, inds: np.ndarray) -> None:
        self.inds = inds
        self.n_elements = len(self.inds)

    def sample(self, n:int = 1) -> np.ndarray:
        ind = np.random.choice(self.n_elements, n, replace=False)
        return self.inds[ind]

    def fit(self, model:Clustering):
        pass
