import csv
import numpy as np
import pickle

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
pids = properties[:, 0].astype(int).tolist()
properties = np.column_stack((properties[:, 4], properties[:, 5], properties[:, 19], properties[:, 26], properties[:, 11], properties[:, 27], properties[:, 40], properties[:, 51], properties[:, 53]))
properties[properties == ''] = 0.0 # Deal with missing elements
properties = properties.astype(float)

pickle.dump(pids, open("data/pids.p", "wb"))
pickle.dump(properties, open("data/properties.p", "wb"))

transactions = dict()
with open("data/train_2016_v2.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        transactions[int(row[0])] = float(row[1])

pickle.dump(transactions, open("data/transactions.p", "wb"))
