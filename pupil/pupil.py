# from pupil.db.vector import VectorDB
# from pupil.db.meta import BaseDB
# from pupil.models.clustering import Clustering
# from typing import Union

# give us some ideas about which data is more important
# model should give us some ideas about how far important each data point is
# spliter : to group the points into some general groups
from pupil.models.clustering import Clustering, Splitter
from nptyping import NDArray
from typing import Any


class PriorityGenerator:
    def __init__(
        self, 
        model: Clustering, 
        spliter: Splitter, 
        center_of_labeled_data: NDArray[(1, Any), Any],
    ):
        ## cc = center of cluster
        self.model = model
        self.spliter = spliter
        self.center_of_labeled_data = center_of_labeled_data
        self.splits = None
   
    def fit(self, embeddings: NDArray[(Any, Any)]):
        self.model.fit(embeddings)
        cc_dist_from_labeled_emb, cc_ind_from_labeled_emb = self.model.distance_to_cluster_centers(self.center_of_labeled_data)
        self.spliter.fit(cc_dist_from_labeled_emb , cc_ind_from_labeled_emb)
        self.splits = self.spliter.splits