import math
from cProfile import label
from typing import Any

import numpy as np
from nptyping import NDArray
from pupil.models import FaissKMeansClustering
from pupil.sampling.base import RandomSampler
from pupil.sampling.cluster_based import ClusteringSampler
from pupil.sampling.uncertainty import UncertaintySampler, entropy_based
from pupil.types import NDArray2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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


class ATLASSampler:
    """
    Active transfer learning for adaptive sampling.
    validation items are predicted by the model and bucketed as “Correct” or
    “Incorrect,” according to whether they were classified correctly.
    The last layer of the model is retrained to predict whether items
    are “Correct” or “Incorrect,” effectively turning the two buckets into new labels.
    We apply the new model to the unlabeled data, predicting whether each item
    will be “Correct” or “Incorrect.” We can sample the most likely to be
    “Incorrect.”
    """

    def __init__(
        self,
        trust_model=None,
        uncertainty_strategy="entropy_based",
    ):
        """_summary_

        Args:
            trust_model (optional): Any Sklean Classifier if None, then will use LogisticRegression.
                Defaults to None.
            uncertainty_strategy (str):
                One of
                ['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based'].
                 Defaults to "entropy_based".
        """
        self.scaler = StandardScaler()
        self.trust_model = trust_model
        if self.trust_model is None:
            self.trust_model = LogisticRegression(
                solver="saga",
                n_jobs=-1,
                penalty="l1",
                max_iter=1000,
            )
        self.unc_sampler = UncertaintySampler.from_strategy(uncertainty_strategy)

    def fit(
        self,
        unlabeled_emb: NDArray2D,
        validation_y_hat_prob: NDArray2D,
        validation_emb: NDArray2D,
        validation_y: np.ndarray,
    ):
        """
        Steps:
            1. Scale embs data
            2. Use `validation_y_hat_prob` to see which class model picked
            3. Compare results of 2 and validation_y and tag them with corr/incorr
            4. Train `trust_model` on this new generated tags from step 3
            5. Use model from step 4 to predict tags for all `unlabeled_emb`
            6. Sample from predictions of 5 and pick select indecies with highest
            chance of `incorrect` tags
        Args:
            unlabeled_emb (NDArray2D): Embs of trained data from trained model
            validation_y_hat_prob (NDArray2D): Predictions of a model on validation set
            validation_emb (NDArray2D): Embs of validation data from trained model
            validation_y (np.ndarray): Actual lables of validation set
        """

        unlabeled_emb = self.scaler.fit_transform(
            unlabeled_emb
        )  # scale input for saga solver
        validation_emb = self.scaler.transform(validation_emb)
        val_y_hat = np.argmax(
            validation_y_hat_prob, axis=1
        )  # change prob to actual preds
        val_y_corr_incorr = np.where(
            validation_y == val_y_hat, "correct", "incorrect"
        )  # label 0 to correct and 1 to incorrect

        self.trust_model.fit(
            validation_emb, val_y_corr_incorr
        )  # train model to predict if it will be corr/incorr
        unlabeled_y_hat_probs = self.trust_model.predict_proba(
            unlabeled_emb
        )  # predict the corr/incorr

        incorrect_filter = (
            np.argmax(unlabeled_y_hat_probs, axis=1) == 1
        )  # select the ones that will be incorrect

        # since we filter, we need a mapper to understand what inds in the filtered array represnt in original array
        inds = np.argwhere(incorrect_filter).flatten().tolist()
        rev_inds_mapper = {i: ind for i, ind in enumerate(inds)}

        incorrect_y_hat_prob = unlabeled_y_hat_probs[
            incorrect_filter
        ]  # probability this would be incorrect
        self.unc_sampler.fit(incorrect_y_hat_prob)

        # select the one models most certain it would be classified incorrectly
        inds_ = self.unc_sampler.indices_.tolist()[::-1]  # type: ignore
        self.indices_ = [*map(rev_inds_mapper.get, inds_)]


if __name__ == "__main__":
    all_data_size = 300
    dim = 60
    n_clusters = 5
    data_in_a_cluster = all_data_size / n_clusters
    train_size = int(data_in_a_cluster * 3.5)

    import pandas as pd
    from sklearn.datasets import make_biclusters
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    data, rows, columns = make_biclusters(
        shape=(all_data_size, dim),
        n_clusters=n_clusters,
        noise=5,
        shuffle=False,
        random_state=0,
    )

    labels = [i // data_in_a_cluster for i in range(all_data_size)]
    rng = np.random.RandomState(0)

    train_val_x = data[:train_size]
    train_val_y = labels[:train_size]
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_val_x, train_val_y, train_size=0.8
    )

    unlabeled_data = data[train_size:]
    clf = LogisticRegression(random_state=0, solver="liblinear").fit(tr_x, tr_y)

    probs = clf.predict_proba(val_x)
    sampler = ATLASSampler()
    sampler.fit(
        unlabeled_emb=unlabeled_data,
        validation_emb=val_x,
        validation_y=np.array(val_y),
        validation_y_hat_prob=probs,
    )
    preds = clf.predict_proba(unlabeled_data)
    preds = np.argmax(preds, axis=1).reshape((-1, 1))
    preds = preds[sampler.indices_]
    actual = labels[train_size:]
    df = pd.DataFrame([preds, actual], columns=["preds", "actual"])
    print(df)
