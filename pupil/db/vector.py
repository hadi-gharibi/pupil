from abc import ABC, abstractmethod
import faiss
import numpy as np
from typing import Tuple, Protocol
from pupil.db.config import FaissConf

class VectorDB(Protocol):
    def train(self,):
        ...

    def search(self):
        ...

    def add(self):
        ...
        
    @property
    def n_elements(self):
        ...


class FaissVectorDB:
    def __init__(
        self, 
        emb_size:np.ndarray, 
        nlist:int = FaissConf.nlist, 
        nprobe:int = FaissConf.nprobe
        )->None:

        quantizer = faiss.IndexFlatL2(emb_size)
        self.index = faiss.IndexIVFFlat(quantizer, emb_size, nlist)
        self.index.nprobe = nprobe

    def train(self, embs: np.ndarray) -> None:
        self.index.train(embs)

    def add(self, emb: np.ndarray) -> None:
        self.index.add(emb)

    def search(self, query , n_results :int = 4) -> Tuple[np.ndarray, np.ndarray]:
        distances, inds = self.index.search(query, n_results) 
        return distances, inds # [(self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls)]
    
    @property
    def n_elements(self):
        return self.index.ntotal