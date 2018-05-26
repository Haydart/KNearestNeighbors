import numpy as np
from sklearn import preprocessing


def load_data(name):
    with open('./datasets' + name + '.csv', 'rb') as file:
        data = np.loadtxt(file, dtype=float, delimiter=',')
        X = data[:, :-1].astype(np.float)
        Y = data[:, -1].astype(np.int)
    return X, Y


def scale_to_min_max(x):
    return preprocessing.MinMaxScaler().fit_transform(x)


def scale_standard(x):
    return preprocessing.StandardScaler().fit_transform(x)


def vote_harmonic_weights(args):
    w = np.ones(args.shape)
    for i in range(w.shape[1]):
        w[:, i] = w[:, i] / (i + 1)
    return w
