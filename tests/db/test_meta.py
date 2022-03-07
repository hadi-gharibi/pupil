import pytest
from pupil.db.meta import PandasDB
import pandas as pd
from typing import Any, Optional, Union, List, Any
from dataclasses import dataclass

@dataclass
class DataSchema:
    id: int
    data: Any
    path: Optional[str]
    label: Optional[Union[str, int]]
    
    def __eq__(self, other):
        return self.id == other.id and self.data == other.data and self.path == other.path  and self.label == other.label 

@dataclass
class DataSchemaNoPass:
    index: int
    data: Any
    path: Optional[str]
    label: Optional[Union[str, int]]


def test_pandasdb_creating():
    to_pass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['index', 'data', 'path', 'label'])
    to_notpass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['noindex', 'data', 'path', 'label'])
    
    
    with pytest.raises(ValueError) as exc_info:
        PandasDB(schema=DataSchema, label='no_label')
    assert exc_info.value.args[0].startswith("no_label must be in your schema.")

    with pytest.raises(ValueError) as exc_info:
        PandasDB(schema=DataSchemaNoPass, label='label')
    assert exc_info.value.args[0] == "Please remove the `index` keyword from your schema"

def test_pandasdb_add():
    to_pass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['id', 'data', 'path', 'label'])

    db = PandasDB(schema=DataSchema, label='label')

    db.add(to_pass)
    assert len(db.df) == 2

def test_len():
    to_pass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['id', 'data', 'path', 'label'])
    db = PandasDB(schema=DataSchema, label='label')
    db.add(to_pass)

    assert len(db) == len(db.df)

def test_pandasdb_get():
    to_pass = pd.DataFrame([
        [1,'test','stat', 'sata'],
        [5, 12, 'end', 'dad']
    ], 
        columns=['id', 'data', 'path', 'label'])
    db = PandasDB(schema=DataSchema, label='label')
    db.add(to_pass)

    assert db.get(0) == [DataSchema(id = 1, data = 'test',path = 'stat', label = 'sata')]
    assert db[0] == [DataSchema(id = 1, data = 'test',path = 'stat', label = 'sata')]

    assert db.get(1) == [DataSchema(id = 5, data = 12,  path='end', label='dad')]