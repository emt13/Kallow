import csv
import numpy as np
import pickle
import sys

''' *************************** Read in the data ****************************'''
properties = list()
with open("properties_2016.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        properties.append(row)

chunk_size = round(len(properties)/10)
properties = np.array(properties[:chunk_size])
pickle.dump(properties, open("properties.p", "wb"))

transactions = list()
with open("train_2016_v2.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        transactions.append(row)

transactions = np.array(transactions)
pickle.dump(transactions, open("transactions.p", "wb"))
