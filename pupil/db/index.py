from abc import ABC, abstractmethod

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

    @abstractmethod
    def add_batch(self):
        pass

