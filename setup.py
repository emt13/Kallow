import csv
import numpy as np
import pickle
import sys

''' *************************** Read in the data ****************************'''
properties = list()
with open("data/properties_2016.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        properties.append(row)

chunk_size = round(len(properties)/10) # Determine chunk size
properties = np.array(properties[:chunk_size]) # Select only a small chunk
# Select only the columns that matter
properties = np.column_stack((properties[:, 4], properties[:, 5], properties[:, 6], properties[:, 9], properties[:, 13], properties[:, 32]))

pickle.dump(properties, open("data/properties.p", "wb"))

transactions = list()
with open("data/train_2016_v2.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        transactions.append(row)

transactions = np.array(transactions)
pickle.dump(transactions, open("data/transactions.p", "wb"))
