from pupil.db.vector import NDArray2D, VectorDB
from pupil.db.meta import MetaDataDB
from typing import Any
from .config import NDArray2D

class DataBase:
    def __init__(self, vecdb: VectorDB, metadb:MetaDataDB):
        
        self.vecdb = vecdb
        self.metadb = metadb

    def add(self, metadata: Any, embeddings: NDArray2D):
        if len(metadata) != len(embeddings):
             raise ValueError(f'You have {len(metadata)} data in your metadata and {len(embeddings)} data in your embeddings. These numbers should be same.')
        self._add_metadata(metadata)
        self._add_embeddings(embeddings)

    def _add_embeddings(self, embeddings):
        self.vecdb.train(embeddings)
        self.vecdb.add(embeddings)

    def _add_metadata(self, data):
        self.metadb.add(data)

    def __getitem__(self, i):
        if len(self.vecdb) != len(self.metadb):
            raise ValueError(f'You have {len(self.vecdb)} data in your VectorDB and {len(self.metadb)} data in your MetaDB. These should be same.')
        return { 
            'embeddings' : self.vecdb[i],
            'metadata' : self.metadb[i]
        }

    def get(self, i, emb=True):
        res = {}
        if emb :
            res['embeddings'] = self.vecdb[i]
        res['metadata'] = self.metadb[i]
        return res

    def embbeding_search(self, embeddings: NDArray2D, n_results):
        _, inds = self.vecdb.search(query = embeddings, n_results=n_results)
        res = [self.metadb[row] for row in inds]
        return res