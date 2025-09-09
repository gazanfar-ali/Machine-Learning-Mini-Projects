from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    mnist = fetch_openml('mnist_784', as_frame=True)
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.int8)

    # Train-test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Shuffle training data
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

    return X_train, X_test, y_train, y_test
