from abc import ABC, abstractmethod
import faiss
import numpy as np
from pupil.models.config import FaissKMeansConfig
from sklearn.cluster import AgglomerativeClustering


class FaissKMeans:
    def __init__(self, 
                n_clusters = FaissKMeansConfig.n_clusters, 
                n_init = FaissKMeansConfig.n_init, 
                max_iter = FaissKMeansConfig.max_iter):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

    def distance_to_cluster_centers(self, X):
        D, I = self.kmeans.index.search(X.astype(np.float32), self.n_clusters)
        return D, I

class Distance1DSplitter:
    def __init__(self, X):
        self.alg = AgglomerativeClustering(n_clusters=3)
        self.alg.fit(X.reshape((-1,1)))

    @property
    def tag_to_index(self):
        tags = ['close', 'mid', 'far']

        inds = np.argwhere(np.diff(self.alg.labels_) != 0).flatten().tolist()
        inds.insert(0, -1)
        inds.append(len(self.alg.labels_))

        tag_dict = {}
        for i,end in enumerate(inds[1:]):
            start = inds[i] +1
            tag_dict[tags[i]] = (start, end+1)
        return tag_dict

