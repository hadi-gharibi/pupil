import math

import numpy as np
from pupil.models import FaissKMeansClustering
from pupil.sampling.base import RandomSampler
from pupil.sampling.cluster_based import ClusteringSampler
from pupil.sampling.uncertainty import UncertaintySampler, entropy_based
from pupil.types import NDArray2D


class ClusteredUncertaintySampler:
    """
    First, you sample the uncertainity_rate% most uncertain items;
    then you apply clustering to ensure diversity within that selection,
    sampling the centroid of each cluster.
    """

    def __init__(
        self, uncertainity_rate=0.5, n_clusters=20, uncertainty_strategy="entropy_based"
    ):
        if not (0 < uncertainity_rate <= 1):
            raise ValueError("uncertainity_rate most be between 0 and 1")
        self.uncertain_rate = uncertainity_rate
        self.unc_sampler = UncertaintySampler.from_strategy(uncertainty_strategy)
        clustering_model = FaissKMeansClustering(n_clusters=n_clusters)
        self.clustering = ClusteringSampler(clustering_model=clustering_model)

    def fit(self, prob_predictions: NDArray2D, unlabeled_data):
        if len(prob_predictions) != len(unlabeled_data):
            raise ValueError(
                "Lenght of prob_predictions must be equal to lenght unlabeled_data"
            )
        self.unc_sampler.fit(prob_predictions)
        uncertain_count = math.ceil(len(unlabeled_data) * self.uncertain_rate)

        inds = self.unc_sampler.indices_[:uncertain_count].tolist()  # type: ignore
        rev_inds_mapper = {i: ind for i, ind in enumerate(inds)}
        filtered_unlabeld_data = unlabeled_data[inds]
        self.clustering.fit(filtered_unlabeld_data)
        self.indices_ = [*map(rev_inds_mapper.get, self.clustering.indices_)]


class HighEntropyClusterSampler:
    """
    If you have high entropy in a certain cluster, a lot of confusion exists about the right
    labels for items in that cluster.
    That this approach works best when you have data with accurate labels and
    are confident that the task can be solved with machine learning.
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.clustering_model = FaissKMeansClustering(n_clusters=n_clusters)

    def fit(self, prob_predictions, unlabeled_data):
        highest_average_uncertainty = 0.0
        most_uncertan_cluster = -1
        self.clustering_model.fit(unlabeled_data)
        _, clusters = self.clustering_model.predict(unlabeled_data)
        clusters = clusters.flatten()
        for cluster_id in range(self.n_clusters):
            this_cluster_prob_dist = prob_predictions[clusters == cluster_id]
            avg_cluster_uncertanity = entropy_based(this_cluster_prob_dist).mean()
            if highest_average_uncertainty < avg_cluster_uncertanity:
                most_uncertan_cluster = cluster_id
        inds = np.argwhere(clusters == most_uncertan_cluster)
        rs = RandomSampler()
        rs.fit(inds)
        self.indices_ = rs.indices_


if __name__ == "__main__":
    all_data_size = 300
    dim = 60
    n_clusters = 5
    data_in_a_cluster = all_data_size / n_clusters
    train_size = int(data_in_a_cluster * 3.5)

    from sklearn.datasets import make_biclusters
    from sklearn.linear_model import LogisticRegression

    data, rows, columns = make_biclusters(
        shape=(all_data_size, dim),
        n_clusters=n_clusters,
        noise=5,
        shuffle=False,
        random_state=0,
    )

    labels = [i // data_in_a_cluster for i in range(all_data_size)]
    rng = np.random.RandomState(0)

    train_data = data[:train_size]
    unlabeled_data = data[train_size:]
    clf = LogisticRegression(random_state=0, solver="liblinear").fit(
        train_data, labels[:train_size]
    )
    probs = clf.predict_proba(unlabeled_data)
    print(probs)
    sampler = HighEntropyClusterSampler(n_clusters=2)
    sampler.fit(prob_predictions=probs, unlabeled_data=unlabeled_data)
    print(sampler.indices_)
