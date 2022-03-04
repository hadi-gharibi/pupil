from abc import ABC, abstractmethod
from cProfile import label
import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class RawData:
    index: int
    data: Any
    path: Optional[str]
    label: Optional[Union[str, int]]

class BaseDB(ABC):
    @abstractmethod
    def add(self, X:RawData):
        pass

    @abstractmethod
    def get(self, index:int):
        pass
    
    def __getitem__(self, i):
        return self.get(i)

class PandasDB(BaseDB):
    def __init__(self, data:pd.DataFrame) -> None:
        if data.columns.tolist() != ['index', 'data', 'path', 'label']:
            raise ValueError("Your DataFrame columns must be `['index', 'data', 'path', 'label']`")
        self.df = pd.DataFrame(data)
        self.df.set_index('index', inplace=True)

    def add(self, data:RawData) -> None:
        data = pd.DataFrame(
            [[data.index, data.data, data.path, data.label]],
            columns=['index', 'data', 'path', 'label']
        ).set_index('index')
        self.df = pd.concat([self.df, data], axis=0)

    def get(self, index) -> RawData:
        return RawData(index=index, **self.df.loc[index].to_dict())