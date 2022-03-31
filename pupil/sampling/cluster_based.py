from __future__ import annotations

import numpy as np
from pupil.models.clustering import Clustering, FaissKMeansClustering
from pupil.types import NDArray2D


class ClusteringSampler:

    """
    Clustering sampling:
    1. Get the closest data to centroids
    2. Get outliers in each cluster
    3. Randomly sample from each cluster
    4. Combine them all

    """

    def __init__(
        self,
        clustering_model: Clustering,
    ) -> None:
        self.clustering_model = clustering_model
        self.indices_ = None

    def _distance_clusters(
        self, n_clusters: int, distances: np.ndarray, clusters: np.ndarray
    ) -> NDArray2D:
        """For eaech cluster number in the list of inds ordered by their distance
        from the center of cluster

        Args:
            n_clusters (int): Number of clusters
            distances (np.ndarray): Distance between each data point and it center of cluster
            classes (np.ndarray): Cluster number fo each data

        Returns:
             NDArray2D: Row number represent the cluster number. In each row for each index you see
             the distance between that data point to the center of the cluster
        """

        cluster_inds = None
        for c in range(n_clusters):
            mask = clusters != c
            cnt = len(mask) - mask.sum()
            masked_array = np.ma.array(distances, mask=mask)
            # inds = masked_array.argsort()[:cnt]
            if cluster_inds is None:
                cluster_inds = masked_array
            else:
                cluster_inds = np.ma.vstack([masked_array, cluster_inds])
            # cluster_inds[c] = inds[:cnt]

        return cluster_inds

    def _get_outliers(self, inds) -> np.ndarray:
        """get the data with the most distance from the center of their clusters

        Args:
            inds (masked ndarray): _description_

        Returns:
            np.ndarray: inds
        """
        cols = ~inds.mask.sum(axis=1)
        return inds[list(range(inds.shape[0])), cols]

    def _order_inds(self, inds):
        """order args inside each clsuter

        Args:
            inds (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind_mask = inds.shape[1] - inds.mask.sum(axis=1)
        mask = np.zeros(inds.shape)
        for row, col in enumerate(ind_mask):
            mask[row, col:] = 1

        inds = np.ma.array(inds.argsort(axis=1), mask=mask)
        return inds

    def _get_centroids(self, inds: np.ndarray) -> np.ndarray:
        """Get closest data point to centeroid

        Args:
            inds (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return inds[:, 0]

    def _get_random(self, inds) -> np.ndarray:
        """Shuffle rest of the data(it removes outliers and centroids), and sample from each cluster

        Args:
            inds (_type_):

        Returns:
            np.ndarray: inds
        """
        inds = inds.copy()
        inds = inds[:, 1:]  # remove centroids
        inds.mask[
            list(range(inds.shape[0])), inds.mask.argmax(1) - 1
        ] = True  # remove outliers
        np.random.shuffle(inds)  # shuffle columns
        inds = inds.T.filled(999)
        inds = inds.flatten()
        return inds[inds != 999]

    def _fit(self, X: NDArray2D):
        """fit the clsutering model

        Args:
            X (NDArray2D):
        """
        self.clustering_model.fit(X)
        dist, clusters = [a.flatten() for a in self.clustering_model.predict(X)]
        return dist, clusters

    def _create_indices(self, dist, clusters):
        cluster_inds = self._distance_clusters(
            self.clustering_model.n_clusters, dist, clusters
        )
        ordered_cluster_inds = self._order_inds(cluster_inds)

        centeroids = self._get_centroids(ordered_cluster_inds).tolist()
        outlier = self._get_outliers(ordered_cluster_inds).tolist()
        rnd = self._get_random(ordered_cluster_inds).tolist()

        return centeroids + outlier + rnd

    def fit(self, X: NDArray2D) -> None:
        dist, clusters = self._fit(X)
        self.indices_ = self._create_indices(dist, clusters)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(  # type: ignore
        n_samples=20, centers=centers, cluster_std=0.7, random_state=42
    )
    c = FaissKMeansClustering(n_clusters=3)
    sampler = ClusteringSampler(clustering_model=c)
    r = sampler.fit(X)
    print(r)
