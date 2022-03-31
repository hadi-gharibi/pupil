from typing import List

import numpy as np


class RandomSampler:
    def __call__(self, inds: np.ndarray) -> List[int]:
        new_inds = list(range(len(inds)))
        np.random.shuffle(new_inds)
        return new_inds
