import pyexpat
import pytest
from pupil.db import Faiss
from pupil.db.config import FaissConf
import numpy as np
import yaml

EMB_SIZE = 64
QUERY_NUMBERS = 100
DATABASE_SAMPLE = 1000

@pytest.fixture
def train_data():
    d = EMB_SIZE                          # dimension
    nb = DATABASE_SAMPLE                      # database size
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    
    return xb

@pytest.fixture
def query_data():
    nq = QUERY_NUMBERS
    xq = np.random.random((nq, EMB_SIZE)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    return xq

@pytest.fixture
def faiss_db():
    conf = FaissConf()
    db = Faiss(emb_size= EMB_SIZE, nlist=conf.nlist, nprobe=conf.nprobe)
    return db

@pytest.fixture
def faiss_trained_db(faiss_db, train_data):
    faiss_db.train(train_data)
    return faiss_db


def test_faiss_train(faiss_db, train_data):
    faiss_db.train(train_data)

def test_faiss_add(faiss_trained_db, train_data):
    faiss_trained_db.add(train_data)

def test_faiss_search(faiss_trained_db, query_data):
    D, I = faiss_trained_db.search(query_data, 5)
    assert len(D) == QUERY_NUMBERS
    assert I.shape == (QUERY_NUMBERS ,5)

