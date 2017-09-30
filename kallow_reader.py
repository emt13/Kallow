import numpy as np
import pickle

def kallow_read():
    properties = pickle.load(open("data/properties.p", "rb"))
    transactions = pickle.load(open("data/transactions.p", "rb"))
    return properties, transactions
