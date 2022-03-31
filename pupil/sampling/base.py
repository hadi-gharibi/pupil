from __future__ import annotations

from typing import List, Protocol, Union

import numpy as np
import numpy.typing as npt


class Sampler(Protocol):
    indices_: Union[List[int], npt.NDArray]


class RandomSampler:
    def __init__(
        self,
    ):
        self.indices_ = None

    def fit(
        self,
        inds: npt.NDArray[np.int32],
    ) -> None:
        new_inds = list(range(len(inds)))
        np.random.shuffle(new_inds)
        self.indices_ = new_inds

    def __call__(self, inds: npt.NDArray[np.int32]) -> List[int]:
        self.fit(inds)
        return self.indices_  # type: ignore
