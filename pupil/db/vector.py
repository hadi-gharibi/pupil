from abc import ABC, abstractmethod
import faiss
import numpy as np
from typing import List
import yaml


class BaseVectorDB(ABC):
    @abstractmethod
    def train_ind(self,):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def add(self):
        pass



class Faiss(BaseVectorDB):
    def __init__(self, emb_size:np.Array, nlist:int, nprobe:int)->None:

        quantizer = faiss.IndexFlatL2(emb_size)
        self.index = faiss.IndexIVFFlat(quantizer, emb_size, nlist)
        self.index.nprobe = nprobe

    def train(self, embs: np.Array) -> None:
        self.train(embs)

    def add(self, emb:np.Array) -> None:
        self.index(emb)

    def search(self, query , n_results :int = 4) -> List[np.Array, np.Array]:
        distances, inds = self.index.search(query, n_results) 
        return distances, inds # [(self.index.ntotal, n_resutls),  (self.index.ntotal, n_resutls)]
