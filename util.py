import numpy as np
import os
import sys
import scipy.io as sio
from sklearn.model_selection import train_test_split


class DataSet(object):
    def __init__(self, data, labels):
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same length")
        self.data = data
        self.labels = labels

    def split(self, train=0.8):
        if train < 0 or train > 1:
            raise ValueError("train has to be in range [0,1]")
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, train_size=train, test_size=1-train)
        train = DataSet(x_train, y_train)
        test = DataSet(x_test, y_test)
        return train, test

def load_data():
    name = "data/var_u"
    d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
    data = d['F']
    y = d['y']
    labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    data_sets = DataSet(data, labels)

    return data_sets





