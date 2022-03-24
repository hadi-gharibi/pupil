import math
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, Tuple, Union

import faiss
import numpy as np
from nptyping import NDArray
from pupil.db.config import FaissConf
from scipy import spatial

from .config import NDArray2D, SimilarityType


class VectorDB(Protocol):
    def search(
        self, query: NDArray2D, n_results: int = 4
    ) -> Tuple[NDArray2D, NDArray2D]:
        """Search embeddings

        Args:
            query (NDArray2D): Vectors to search
            n_results (int, optional): Number of results per query. Defaults to 4.

        Returns:
            Tuple[NDArray2D, NDArray2D]: Return (Distances, indices)
        """
        ...

    def build_index(self, embeddings: NDArray2D) -> None:
        """Index the data

        Args:
            embeddings (NDArray2D): _description_
        """
        ...

    def add(self, embeddings: NDArray2D) -> None:
        """Add embeddings into the database

        Args:
            embeddings (NDArray2D):
        """
        ...

    def __len__(self):
        ...

    def __getitem__(self, i):
        ...


class FaissVectorDB:
    def __init__(
        self,
        similarity_metric: SimilarityType = FaissConf.similarity_type,
        nlist: Optional[int] = FaissConf.nlist,
        nprobe: Optional[int] = FaissConf.nprobe,
    ) -> None:

        self.similarity_metric = similarity_metric
        self.nlist = nlist
        self.nprobe = nprobe

    def build_index(self, embeddings: NDArray2D) -> None:

        size, dim = embeddings.shape
        if not self.nlist:
            self.nlist = min(4096, 8 * round(math.sqrt(size)))

        if size < 4 * 10000:
            fac_str = "Flat"
        elif size < 80 * 10000:
            fac_str = "IVF" + str(self.nlist) + ",Flat"
        elif size < 200 * 10000:
            fac_str = "IVF16384,Flat"
        else:
            fac_str = "IVF16384,PQ8"

        self.index = faiss.index_factory(dim, fac_str, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = min(self.nprobe, self.nlist)  # type: ignore

        if not self.index.is_trained:
            self.index.train(embeddings)
        if self.similarity_metric == SimilarityType.COSINE:
            faiss.normalize_L2(embeddings)
        self.add(embeddings)

    def __getitem__(self, i: Union[int, List[int], NDArray[(Any,), np.int32]]):
        if isinstance(i, slice):
            start = i.start
            if not start:
                start = 0

            stop = i.stop
            if not stop:
                stop = self.__len__()

            step = i.step
            if not step:
                step = 1

            i = list(range(start, stop, step))

        if not self.index.is_trained:
            raise ValueError("First add data to the database.")

        if type(i) is not int:
            return np.vstack([self.__getitem__(ind) for ind in i])  # type: ignore

        if hasattr(self.index, "make_direct_map"):
            self.index.make_direct_map()
        return self.index.reconstruct(i).reshape(1, -1).astype(np.float32)

    def add(self, embeddings: NDArray2D) -> None:
        self.index.add(embeddings)

    def search(
        self, query: NDArray2D, n_results: int = 4
    ) -> Tuple[NDArray2D, NDArray2D]:
        if self.similarity_metric == SimilarityType.COSINE:
            faiss.normalize_L2(query)
        distances, inds = self.index.search(query, n_results + 1)
        return (
            distances[:, 1:],
            inds[:, 1:],
        )  # ((self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls))

    def __len__(
        self,
    ):
        return self.index.ntotal
