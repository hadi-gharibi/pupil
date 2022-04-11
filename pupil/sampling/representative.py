import numpy as np
from pupil.models.clustering import FaissKMeansClustering


class RepresentativeSampler:
    """
    Cluster your training data and your unlabeled data independently,
    identify the clusters that are most representative of your
    unlabeled data, and oversample from them.
    This approach gives you a more diverse set of items than
    representative sampling alone
    """

    def __init__(self, n_clusters):
        self.clustering_training_model = FaissKMeansClustering(n_clusters)
        self.clustering_unlabeld_model = FaissKMeansClustering(n_clusters)

    def _fit(self, training_data, unlabeled_data):
        self.clustering_training_model.fit(training_data)
        self.clustering_unlabeld_model.fit(unlabeled_data)

    def _dist_to_cluster_center(self, model, data):
        dists, _ = model.distance_to_cluster_centers(data)
        return dists[:, 0]

    def representativeness_score(self, unlabeled_data):
        dist_to_cent_training_data = self._dist_to_cluster_center(
            self.clustering_training_model, unlabeled_data
        )

        dist_to_cent_unlabled_data = self._dist_to_cluster_center(
            self.clustering_unlabeld_model, unlabeled_data
        )

        representativeness = dist_to_cent_unlabled_data - dist_to_cent_training_data

        return representativeness

    def fit(self, training_data, unlabeled_data):
        self._fit(training_data, unlabeled_data)
        scores = self.representativeness_score(unlabeled_data)
        self.indices_ = np.argsort(
            scores,
        )


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    centers = [[2, 2], [-2, -2]]
    train, labels_true = make_blobs(  # type: ignore
        n_samples=50, centers=centers, cluster_std=0.1, random_state=42
    )

    centers = [[2, 2], [-2, 2]]
    test, labels_true = make_blobs(
        n_samples=10, centers=centers, cluster_std=0.1, random_state=42
    )
    print("teeee")
    print(labels_true)
    sampler = RepresentativeSampler(n_clusters=2)
    sampler.fit(train, test)
    print(sampler.indices_)
