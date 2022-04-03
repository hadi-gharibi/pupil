from __future__ import annotations

from typing import Any, Callable, List, Optional, TypedDict, Union

from nptyping import NDArray
from pandas import DataFrame
from pupil.db.meta import MetaDataDB, PandasDB
from pupil.db.vector import FaissVectorDB, VectorDB
from pupil.types import IsDataclass, NDArray2D


class GetDatabase(TypedDict):
    metadata: Optional[List[IsDataclass]]
    embeddings: Optional[NDArray2D]


class DataBase:
    def __init__(self, vecdb: VectorDB, metadb: MetaDataDB):

        self.vecdb = vecdb
        self.metadb = metadb

    @classmethod
    def from_encoder(
        cls,
        data_schema: IsDataclass,
        label_col_name: str,
        data_col_name: str,
        data: DataFrame,
        encoder: Callable[..., NDArray2D],
    ):
        vecdb = FaissVectorDB()
        metadb = PandasDB(schema=data_schema, label=label_col_name)
        kls = cls(vecdb=vecdb, metadb=metadb)
        kls.add(metadata=data, embeddings=encoder(data[data_col_name].tolist()))
        return kls

    def add(self, metadata: Any, embeddings: NDArray2D):
        if len(metadata) != len(embeddings):  # type: ignore
            raise ValueError(
                f"You have {len(metadata)} data in your metadata and {len(embeddings)} data in your embeddings. These numbers should be same."  # type: ignore
            )
        self._add_metadata(metadata)
        self._add_embeddings(embeddings)

    def _add_embeddings(self, embeddings):
        self.vecdb.build_index(embeddings)  # type: ignore

    def _add_metadata(self, data):
        self.metadb.add(data)

    def __getitem__(self, i):
        if len(self.vecdb) != len(self.metadb):  # type: ignore
            raise ValueError(
                f"You have {len(self.vecdb)} data in your VectorDB and {len(self.metadb)} ",
                "data in your MetaDB. These should be same.",  # type: ignore
            )
        return {"embeddings": self.vecdb[i], "metadata": self.metadb[i]}

    def get(
        self,
        i: Union[int, List[int], NDArray[(Any,), Int32]],  # type: ignore
        return_embeddings: bool = False,
        return_metadata: bool = True,
    ) -> GetDatabase:
        res: GetDatabase = {"embeddings": None, "metadata": None}
        if return_embeddings:
            res["embeddings"] = self.vecdb[i]
        if return_metadata:
            res["metadata"] = self.metadb[i]
        return res

    def embbeding_search(
        self, embeddings: NDArray2D, n_results: int
    ) -> NDArray2D:  # type: ignore
        _, inds = self.vecdb.search(query=embeddings, n_results=n_results)
        return inds
