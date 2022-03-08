from cProfile import label
import marshmallow_dataclass
import pandas as pd
from dataclasses import dataclass
from typing import Any, Protocol, Union, List, Any
import numpy as np
from nptyping import NDArray, Int32
import marshmallow_dataclass
from marshmallow.schema import Schema
from .config import IsDataclass


class MetaDataDB(Protocol):
    schema: Schema
    label: str

    def __init__(self, schema: IsDataclass, label: str):
        ...

    def add(self, X: Any):
        ...

    def get(self, index:Union[int, List[int], NDArray[(Any,), Int32]]) -> List[IsDataclass]:
        ...
    def __getitem__(self, i: Union[int, List[int], NDArray[(Any,), Int32]]):
        ...

    def __len__(self):
        ...

    def set_label(self, i: int, input:Any):
        ...

class PandasDB:
    def __init__(self, schema: IsDataclass, label: str) -> None:
        self.schema = marshmallow_dataclass.class_schema(schema)(many=True) # type: ignore
        if label not in self.schema.fields.keys(): 
            raise ValueError(f"{label} must be in your schema. your schema has {[k for k in self.schema.fields.keys()]}")
        if 'index' in self.schema.fields.keys(): 
            raise ValueError('Please remove the `index` keyword from your schema')
        self.label = label
        self.df:pd.DataFrame = None

    def add(self, data:pd.DataFrame) -> None:
        if (self.df is None) and (isinstance(data, pd.DataFrame)):
            if self.label not in (set(data.columns.tolist())):
                raise ValueError(f"Your DataFrame columns must have `{self.label}`")
            self.df = pd.DataFrame(data)

        elif (self.df is not None) and (isinstance(data, pd.DataFrame)):
            self.df = pd.concat([self.df, data], axis=1)

    def get(self, index:Union[int, List[int], NDArray[(Any,), Int32]]) -> List[IsDataclass]:
        
        if isinstance(index, (list, np.ndarray)):
            data = [self.get(i)[0] for i in index]
        else:
            data = self.schema.load( [self.df.iloc[index].to_dict()])
        
        return data

    def set_label(self, i, inp):
        self.df.iloc[i, self.df.columns.get_loc(self.label)] = inp
    
    def __getitem__(self, i: Union[int, List[int], NDArray[(Any,), Int32]]):
        return self.get(i)

    def __len__(self):
        return len(self.df)