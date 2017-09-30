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

properties = np.array(properties)
chunk_size = round(len(properties)/10)
for i in range(0, len(properies), chunk_size):
    pickle.dump(properties[i:i+chunk_size, :], open("properties" + str(i) + ".p", "wb"))

transactions = list()
with open("train_2016_v2.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader) # Skip header
    for row in reader:
        transactions.append(row)

transactions = np.array(transactions)
pickle.dump(transactions, open("train.p", "wb"))
print(transactions.shape)
print(transactions[0])
