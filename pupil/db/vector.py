from abc import ABC, abstractmethod
from ast import Raise
from optparse import Option
import faiss
import numpy as np
from typing import Tuple, Protocol, Any, NewType, Optional
from pupil.db.config import FaissConf
from nptyping import NDArray

NDArray2D = NewType("NDArray2D", NDArray[(Any, Any), Any]) # type: ignore

class VectorDB(Protocol):
    def train(self,):
        ...

    def search(self):
        ...

    def add(self):
        ...
    
    def __len__(self):
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

    def __getitem__(self, i):
        if self.embeddings is None:
            raise ValueError("First add data to the database.")
        return self.embeddings[i]

    def add(self, emb: NDArray2D) -> None:
        self.index.add(emb)

    def search(self, query: NDArray2D, n_results :int = 4) -> Tuple[NDArray2D, NDArray2D]:
        distances, inds = self.index.search(query, n_results + 1) 
        return distances[0, 1:], inds[0, 1:] # ((self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls))

    def __len__(self,):
        return self.index.ntotal