import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings
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

def main():
    print("Loading data...")
    transactions = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])
    test_transactions = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'])
    properties = pd.read_csv('data/properties_2016.csv')
    print("Done.")
    print("-------------------------------------------------------------------")

    print("Parsing data...")
    merged = pd.merge(transactions, properties, on="parcelid", how="left")
    test_merged = pd.merge(test_transactions, properties, on="parcelid", how="left")

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
    min_error = 10000
    best_model = None

    # Tune the hyperparameters
    try:
        for n_est in np.arange(50, 250, 50):
            validation_error = list()
            for depth in np.arange(10, 50, 10):
                # Design the model
                model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1, max_depth=depth, random_state=0, loss='ls')
                model.fit(X_train, Y_train) # Learn the model

                # Evaluate the model
                error = mean_squared_error(Y_validate, model.predict(X_validate))
                if error < min_error:
                    min_error = error
                    best_model = model

                validation_error.append(error)

            plt.plot(np.arange(10, 50, 10), validation_error, label="Number of estimators: " + str(n_est))
    except KeyboardInterrupt:
        pass

    plt.legend()
    plt.savefig('validation.png')

    print("MSE on test data:", mean_squared_error(Y_test, best_model.predict(X_test)))
    print("Test Score:", best_model.score(X_test, Y_test))
    print("Done.")
    print("-------------------------------------------------------------------")

if __name__ == "__main__":
    main()