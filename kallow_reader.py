import numpy as np
import pickle

def kallow_read():
    properties = pickle.load(open("properties.p", "rb"))
    transactions = pickle.load(open("transactions.p", "rb"))
    return properties, transactions
