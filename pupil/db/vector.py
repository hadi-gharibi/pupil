import math
from typing import Optional, Protocol, Sequence, Tuple, Union

import faiss
import numpy as np
from pupil.types import Distance, NDArray2D


class VectorDB(Protocol):
    """Vector database to save the embeddings and fast distance calculation.
    Normal usage: 
        1. Create object
        2. `build_index`
        3. `add` embeddings
    """
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

    def __len__(self) -> int:
        ...

    def __getitem__(self, i: Union[int, Sequence[int], slice]) -> np.ndarray:
        ...


class FaissVectorDB:
    def __init__(
        self,
        similarity_metric: Distance = Distance.COSINE,
        nlist: Optional[int] = None ,
        nprobe: Optional[int] = 5,
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
        if self.similarity_metric == Distance.COSINE:
            faiss.normalize_L2(embeddings)
        self.add(embeddings)

    def __getitem__(self, i: Union[int, Sequence[int], slice]) -> np.ndarray:
        if isinstance(i, slice):
            i = list(range(*i.indices(self.__len__()))) # type: ignore

        if not self.index.is_trained:
            raise ValueError("First add data to the database.")

        if not isinstance(i, int):
            return np.vstack([self.__getitem__(ind) for ind in i])

        if hasattr(self.index, "make_direct_map"):
            self.index.make_direct_map()
        return self.index.reconstruct(i).reshape(1, -1).astype(np.float32)

    def add(self, embeddings: NDArray2D) -> None:
        """Add embeddings into the database

        Args:
            embeddings (NDArray2D):
        """
        self.index.add(embeddings)

    def search(
        self, query: NDArray2D, n_results: int = 4
    ) -> Tuple[NDArray2D, NDArray2D]:
        """Search embeddings to get the closest embeddings to the queries.

        Args:
            query (NDArray2D): Vectors to search
            n_results (int, optional): Number of results per query. Defaults to 4.

        Returns:
            Tuple[NDArray2D, NDArray2D]: Return (Distances, indices)
        """
        if self.similarity_metric == Distance.COSINE:
            faiss.normalize_L2(query)
        distances, inds = self.index.search(query, n_results + 1)
        return (
            distances[:, 1:],
            inds[:, 1:],
        )  # ((self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls))

    def __len__(
        self,
    ) -> int:
        return self.index.ntotal
