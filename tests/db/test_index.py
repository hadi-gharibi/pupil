import pytest
from pupil.db.index import PandasDB, RawData
import pandas as pd

def test_pandasdb_creating():
    to_pass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['index', 'data', 'path', 'label'])
    to_notpass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['noindex', 'data', 'path', 'label'])

    assert PandasDB(to_pass)
    with pytest.raises(ValueError) as exc_info:   
        PandasDB(to_notpass)
    assert exc_info.value.args[0] == "Your DataFrame columns must be `['index', 'data', 'path', 'label']`"

def test_pandasdb_add():
    to_pass = pd.DataFrame([[1,None,'stat', 'sata'],[5, 12, 'end', 'dad']], columns=['index', 'data', 'path', 'label'])
    db = PandasDB(to_pass)
    data = RawData(index=123, data=None, path='./', label=2)
    db.add(data)
    assert len(db.df) == 3

def test_pandasdb_get():
    to_pass = pd.DataFrame([
        [1,'test','stat', 'sata'],
        [5, 12, 'end', 'dad']
    ], 
        columns=['index', 'data', 'path', 'label'])
    db = PandasDB(to_pass)
    data = RawData(index=123, data='test', path='./', label=2)
    db.add(data)

    assert db.get(1) == RawData(index = 1, data = 'test',path = 'stat', label = 'sata')
    assert db[1] == RawData(index = 1, data = 'test',path = 'stat', label = 'sata')

    assert db.get(123) == RawData(index = 123, data = 'test',  path='./', label=2)