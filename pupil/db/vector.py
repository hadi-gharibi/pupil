from abc import ABC, abstractmethod
import faiss
import numpy as np
from typing import List

class BaseVectorDB(ABC):
    @abstractmethod
    def train(self,):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def add(self):
        pass



class Faiss(BaseVectorDB):
    def __init__(self, emb_size:np.ndarray, nlist:int, nprobe:int)->None:

        quantizer = faiss.IndexFlatL2(emb_size)
        self.index = faiss.IndexIVFFlat(quantizer, emb_size, nlist)
        self.index.nprobe = nprobe

    def train(self, embs: np.ndarray) -> None:
        self.index.train(embs)

    def add(self, emb: np.ndarray) -> None:
        self.index.add(emb)

    def search(self, query , n_results :int = 4) -> List[np.ndarray]:
        distances, inds = self.index.search(query, n_results) 
        return distances, inds # [(self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls)]
