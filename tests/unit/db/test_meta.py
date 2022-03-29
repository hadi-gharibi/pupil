from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd
import pytest
from pupil.db.meta import PandasDB
from pytest_lazyfixture import lazy_fixture


@dataclass
class DataSchema:
    index: int
    data: Any
    path: Optional[str]
    label: Optional[Union[str, int]]

    def __eq__(self, other):
        return (
            self.index == other.index
            and self.data == other.data
            and self.path == other.path
            and self.label == other.label
        )


# Data fixtures
@pytest.fixture
def data():
    return [[1, "check1_data", "stat", "sata"], [5, 12, "end", "dad"]]


@pytest.fixture
def good_df(data):
    return pd.DataFrame(data, columns=["index", "data", "path", "label"])


@pytest.fixture
def bad_df(data):
    return pd.DataFrame(data, columns=["noindex", "data", "path", "nolabel"])


# PandasDB Fxitures
@pytest.fixture
def pandasdb():
    return PandasDB(schema=DataSchema, label="label")


@pytest.fixture
def full_pandasdb(pandasdb, good_df):
    pandasdb.add(good_df)
    return pandasdb


@pytest.mark.parametrize("metadb", [PandasDB])
def test_init(metadb):
    with pytest.raises(ValueError) as exc_info:
        metadb(schema=DataSchema, label="fake_label")
    assert exc_info.value.args[0].startswith("fake_label must be in your schema.")

    try:
        metadb(schema=DataSchema, label="label")
    except Exception as e:
        assert False, f"Init ing PandasDB raised and exception {e}"


@pytest.mark.parametrize(
    "metadb",
    [
        (lazy_fixture("pandasdb")),
    ],
)
def test_add(metadb, good_df, bad_df):
    with pytest.raises(ValueError) as exc_info:
        metadb.add(["Non", "Datafram", "object"])  # type: ignore
    assert exc_info.value.args[0] == "Data need to be a DataFrame"

    try:
        metadb.add(good_df)
    except Exception as e:
        assert False, f"Adding df to PandasDB raised and exception {e}"

    with pytest.raises(ValueError) as exc_info:
        metadb.add(bad_df)
    assert exc_info.value.args[0].startswith("Your DataFrame columns must have")


@pytest.mark.parametrize(
    "full_metadb",
    [
        (lazy_fixture("full_pandasdb")),
    ],
)
def test_len(full_metadb):
    assert len(full_metadb) == 2


@pytest.mark.parametrize(
    "full_metadb",
    [
        (lazy_fixture("full_pandasdb")),
    ],
)
def test_get(full_metadb, data):

    assert full_metadb.get(0) == [DataSchema(*data[0])]  # check single index
    assert full_metadb.get([0, 1]) == [
        DataSchema(*data[0]),
        DataSchema(*data[1]),
    ]  # check list as iter
    assert full_metadb[0] == [DataSchema(*data[0])]
    assert full_metadb[:1] == [DataSchema(*data[0])]
    assert full_metadb[1:1] == []
    assert full_metadb[0:2] == [DataSchema(*data[0]), DataSchema(*data[1])]
    assert full_metadb[0:] == [DataSchema(*data[0]), DataSchema(*data[1])]
    assert full_metadb[:2] == [DataSchema(*data[0]), DataSchema(*data[1])]


@pytest.mark.parametrize(
    "full_metadb",
    [
        (lazy_fixture("full_pandasdb")),
    ],
)
def test_filter(full_metadb, data):
    with pytest.raises(ValueError) as exc_info:
        full_metadb.filter("fake_column", "WhatEver")
    assert exc_info.value.args[0].endswith("is not in the DataFrame columns")
    assert full_metadb.filter("index", 1) == [DataSchema(*data[0])]
    assert full_metadb.filter("data", "check1_data") == [DataSchema(*data[0])]
    assert full_metadb.filter("data", 12) == [DataSchema(*data[1])]
