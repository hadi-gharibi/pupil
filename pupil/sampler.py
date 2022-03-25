from typing import Union, Optional, Any
import numpy as np
from pupil.models.clustering import Clustering
from abc import ABC, abstractmethod
from nptyping import NDArray

class Sampler(ABC):
    def __init__(self, inds: NDArray[(Any, ), Any]):
        self._to_pop: NDArray[(Any, ), Any]
        self.inds = inds
        self.strategy()

    def sample(self, n:int = 1) -> NDArray[(1, ), np.int32]:
            if self._to_pop.size == 0:
                raise StopIteration("You already sampled all the data")
            first, self._to_pop = self._to_pop[:n], self._to_pop[n:] # type: ignore
            return first

    @abstractmethod
    def strategy(self):
        # need to create self._to_pop with the right order. 0 index will sample first
        ...

class RandomArraySampler(Sampler):
    def __init__(self, inds: np.ndarray) -> None:
        self._to_pop: np.ndarray
        super().__init__(inds = inds)

    def strategy(self):
        np.random.shuffle(self.inds)
        self._to_pop = self.inds.copy()
