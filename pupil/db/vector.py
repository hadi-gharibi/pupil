from abc import ABC, abstractmethod
import faiss
from typing import Tuple, Protocol, Optional, Union, Any
import numpy as np
from pupil.db.config import FaissConf
from .config import NDArray2D
from nptyping import NDArray

class VectorDB(Protocol):
    def train(self, embeddings: NDArray2D) -> None:
        """In case you need to train the index

        Args:
            embeddings (NDArray2D):
        """
        ...

    def search(self, query: NDArray2D, n_results :int = 4) -> Tuple[NDArray2D, NDArray2D]:
        """Search embeddings

        Args:
            query (NDArray2D): Vectors to search
            n_results (int, optional): Number of results per query. Defaults to 4.

        Returns:
            Tuple[NDArray2D, NDArray2D]: Return (Distances, indices)
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
        emb_size:int, 
        nlist:int = FaissConf.nlist, 
        nprobe:int = FaissConf.nprobe
        )->None:

        self.emb_size = emb_size
        quantizer = faiss.IndexFlatL2(emb_size)
        self.index = faiss.IndexIVFFlat(quantizer, emb_size, nlist)
        self.index.nprobe = nprobe
        self.embeddings:Optional[NDArray2D] = None

    def train(self, embeddings: NDArray2D) -> None:
        self.embeddings = embeddings
        self.index.train(embeddings)

    def __getitem__(self, i: Union[int, NDArray[(Any, ), np.int32]]):
        if self.embeddings is None:
            raise ValueError("First add data to the database.")
        return self.embeddings[i]

    def add(self, emb: NDArray2D) -> None:
        self.index.add(emb)

    def search(self, query: NDArray2D, n_results :int = 4) -> Tuple[NDArray2D, NDArray2D]:
        distances, inds = self.index.search(query, n_results + 1) 
        return distances[:, 1:], inds[:, 1:] # ((self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls))

    def __len__(self,):
        return self.index.ntotal