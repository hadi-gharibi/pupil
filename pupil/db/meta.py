from collections.abc import Iterable as abc_iterator
from typing import Any, Iterable, List, Protocol, Union

import marshmallow_dataclass
import pandas as pd
from marshmallow.schema import Schema
from pupil.types import IsDataclass


class MetaDataDB(Protocol):
    schema: Schema
    label: str

    def __init__(self, schema: IsDataclass, label: str) -> None:
        ...

    def add(self, data: Any):
        ...

    def get(self, index: Union[int, Iterable[int]]) -> List[IsDataclass]:
        ...

    def __getitem__(self, index: Union[int, Iterable[int], slice]) -> List[IsDataclass]:
        """Getting data with row number

        Args:
            i (Union[int, Iterable[int]]): Row numbers

        Returns:
            List[IsDataclass]: List of schema objects
        """
        ...

    def __len__(self) -> int:
        """Lenght of data

        Returns:
            int: _description_
        """
        ...

    def set_label(self, i: int, input: Any) -> None:
        ...


class PandasDB:
    def __init__(self, schema: IsDataclass, label: str) -> None:
        """_summary_

        Args:
            schema (IsDataclass): Dataclass that should describe the schema of data
            label (str): Which column in your DataFrame is the label of data


        """
        self.schema = marshmallow_dataclass.class_schema(schema)(many=True)  # type: ignore
        if label not in self.schema.fields.keys():
            raise ValueError(
                f"{label} must be in your schema. your schema has {[k for k in self.schema.fields.keys()]}"
            )
        self.label = label
        self.df = pd.DataFrame()

    def add(self, data: pd.DataFrame) -> None:
        """Add DataFrame to the database. It should contain the label column you passed to create the instance.

        Args:
            data (pd.DataFrame):

        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data need to be a DataFrame")
        elif self.label not in (set(data.columns.tolist())):
            raise ValueError(f"Your DataFrame columns must have `{self.label}`")

        elif self.df.empty:
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.concat([self.df, data], axis=1)

    def get(self, index: Union[int, Iterable[int]]) -> List[IsDataclass]:
        """Get data from DataFram

        Args:
            index (Union[int, Iterable[int]]): Row number

        Returns:
            List[IsDataclass]: List of schema objects
        """
        return self.__getitem__(index)

    def set_label(self, i: int, input: Any) -> None:
        self.df.iloc[i, self.df.columns.get_loc(self.label)] = input

    def __getitem__(self, index: Union[int, Iterable[int], slice]) -> List[IsDataclass]:
        if isinstance(index, slice):
            index = list(range(*index.indices(self.__len__())))  # type: ignore

        if isinstance(index, abc_iterator):
            data = [self.__getitem__(i)[0] for i in index]
        else:
            data = self.df.iloc[index]
            data = self.schema.load([data.to_dict()])
        return data  # type: ignore

    def __setitem__(self, index: Union[int, Iterable[int], slice], value) -> None:
        if isinstance(index, slice):
            index = list(range(*index.indices(self.__len__())))  # type: ignore

        if isinstance(index, abc_iterator):
            [self.__setitem__(i, value) for i in index]
        else:
            self.set_label(index, value)

    def __len__(self) -> int:
        return len(self.df)

    def filter(self, field: str, value: Any) -> List[IsDataclass]:
        """Filter data base on columns and values

        Args:
            field (str): Name of the column
            value (str): Value to filter on

        Raises:
            ValueError: if field not in the columns

        Returns:
            List[IsDataclass]: List of schema objects
        """
        if field not in (set(self.df.columns.tolist())):
            raise ValueError(f"{field} is not in the DataFrame columns")
        mask = self.df[field] == value
        inds = self.df[mask].index.to_list()
        return self.get(inds)

    @property
    def is_labeled(
        self,
    ):
        return ~self.df[self.label].isna()
