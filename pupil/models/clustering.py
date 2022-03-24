from abc import ABC, abstractmethod
from typing import Dict, Protocol, Tuple

import faiss
import numpy as np
from pupil.db.config import NDArray2D
from pupil.models.config import FaissKMeansConfig
from sklearn.cluster import AgglomerativeClustering


class Clustering(Protocol):
    def fit(self, X: NDArray2D):
        ...

    def predict(self, X: NDArray2D) -> NDArray2D:

        ...

    def distance_to_cluster_centers(self, X: NDArray2D) -> Tuple[NDArray2D, NDArray2D]:
        """After having the center of your clusters, you can use this function to see the distance from X and center of all clusters

        Args:
            X (NDArray2D): The input to check.

        Returns:
            Tuple[NDArray2D, NDArray2D]: Return (Distances, cluster_ids). Shape of each: (#queries, #clusters)
        """
        ...


class FaissKMeansClustering:
    def __init__(
        self,
        n_clusters: int = FaissKMeansConfig.n_clusters,
        n_init: int = FaissKMeansConfig.n_init,
        max_iter: int = FaissKMeansConfig.max_iter,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: NDArray2D) -> None:
        self.kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=self.n_clusters,
            niter=self.max_iter,
            nredo=self.n_init,
        )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X: NDArray2D) -> NDArray2D:
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

    def distance_to_cluster_centers(self, X: NDArray2D) -> Tuple[NDArray2D, NDArray2D]:
        D, I = self.kmeans.index.search(X.astype(np.float32), self.n_clusters)
        return D, I


class Splitter(Protocol):
    def fit(self, X: NDArray2D, clsuter_inds: NDArray2D):
        ...

    @property
    def splits(
        self,
    ):
        ...


class Distance1DSplitter:
    def __init__(self, nsplits=3):
        self.nsplits = nsplits

    def fit(self, X: NDArray2D, clsuter_inds: NDArray2D) -> None:
        self.clsuter_inds = clsuter_inds
        self.alg = AgglomerativeClustering(n_clusters=self.nsplits)
        self.alg.fit(X.reshape((-1, 1)))
        self._tag_to_index_dict = self._tag_to_index()

    def _tag_to_index(self) -> Dict[str, Tuple[int, int]]:
        tags = ["priority_" + str(i) for i in range(self.nsplits)]

        inds = np.argwhere(np.diff(self.alg.labels_) != 0).flatten().tolist()
        inds.insert(0, -1)
        inds.append(len(self.alg.labels_))

        tag_dict = {}
        for i, end in enumerate(inds[1:]):
            start = inds[i] + 1
            tag_dict[tags[i]] = (start, end + 1)
        return tag_dict

    @property
    def splits(self):
        res = {}
        for k, v in self._tag_to_index_dict.items():
            res[k] = self.clsuter_inds[0][v[0] : v[1]]
        return res
