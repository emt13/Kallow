import numpy as np
import pickle

def kallow_read():
    pids = pickle.load(open("data/pids.p", "rb"))
    properties = pickle.load(open("data/properties.p", "rb"))
    transactions = pickle.load(open("data/transactions.p", "rb"))
    return pids, properties, transactions
