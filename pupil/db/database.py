from pupil.db.vector import VectorDB
from pupil.db.meta import MetaDataDB


class DataBase:
    def __init__(self, vecdb: VectorDB, metadb:MetaDataDB):
        
        self.vecdb = vecdb
        self.metadb = metadb

    def add_embeddings(self, embeddings):
        self.vecdb.train(embeddings)
        self.vecdb.add(embeddings)

    def add_metadata(self, data):
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