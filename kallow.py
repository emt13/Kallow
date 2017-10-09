import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import copy
from multiprocessing import Pool, Lock, Manager
from functools import partial
warnings.filterwarnings("ignore")

def imputate_data(df):
    df.dropna(axis=1, how='all', inplace=True)

    df1 = df.filter(regex='id')
    df1.fillna(df1.mode().iloc[0], axis=0, inplace=True)

    df.fillna(df.mean(), axis=0, inplace=True)
    df[list(df1.keys())] = df1

def encode_objects(df):
    for f in df.columns:
        if df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))

def split_merged(df):
    Y = df.logerror.values
    X = df.drop(["parcelid", "transactiondate", "logerror"], axis=1)
    return X, Y

def train(lock, args):
    n = args[0]
    d = args[1]
    r = args[2]
    X_train = args[3]
    Y_train = args[4]
    X_validate = args[5]
    Y_validate = args[6]
    model = GradientBoostingRegressor(n_estimators=n, learning_rate=r, max_depth=d, random_state=0, loss='ls', verbose=1)
    model.fit(X_train, Y_train) # Learn the model

    # Evaluate the model
    error = mean_squared_error(Y_validate, model.predict(X_validate))
    pickle.dump(model, open("model" + str(n) + "-" + str(d) + "-" + str(r) + ".p", "wb+"))

    lock.acquire()
    try:
        print(error, n, d, r, mean_squared_error(Y_train, model.predict(X_train)))
    finally:
        lock.release()

def main():
    print("Loading data...")
    transactions = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])
    test_transactions = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'])
    properties = pd.read_csv('data/properties_2016.csv')
    test_properties = pd.read_csv('data/properties_2017.csv')
    print("Done.")
    print("-------------------------------------------------------------------")

    print("Parsing data...")
    merged = pd.merge(transactions, properties, on="parcelid", how="left")
    test_merged = pd.merge(test_transactions, test_properties, on="parcelid", how="left")

    imputate_data(merged)
    imputate_data(test_merged)

    encode_objects(merged)
    encode_objects(test_merged)

    X, Y = split_merged(merged)
    X_test, Y_test = split_merged(test_merged)

    split_index = round(len(X)*0.7) # 70-30 training/validation split
    X_train, Y_train, X_validate, Y_validate = (X[:split_index], Y[:split_index],
                                                X[split_index:], Y[split_index:])
    print("Done.")
    print("-------------------------------------------------------------------")

    print("Training...")

    p = Pool(20)
    m = Manager()
    l = m.Lock()
    args = list()
    for n_est in np.arange(10, 150, 10):
        for depth in np.arange(5, 30, 5):
            for rate in np.arange(0.001, 0.1, 0.01):
                args.append([n_est, depth, rate, X_train, Y_train, X_validate, Y_validate])

    func = partial(train, l)
    p.map(func, args)

if __name__ == "__main__":
    main()
