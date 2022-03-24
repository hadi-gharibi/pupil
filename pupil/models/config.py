from dataclasses import dataclass

@dataclass
class FaissKMeansConfig:
    n_clusters: int = 8
    n_init: int = 10
    max_iter: int = 100