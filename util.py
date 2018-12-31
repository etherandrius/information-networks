import numpy as np
import os
import sys
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import tensorflow as tf


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


def load_data(data_set, train_size):
    if data_set == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = np.concatenate((x_train, x_test))
        labels = np.concatenate((y_train, y_test))

        data = data.reshape(data.shape[0], 28 * 28)
        data = data.astype('float32') / 255
        labels = tf.keras.utils.to_categorical(labels)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size)
        return (x_train, y_train), (x_test, y_test), 10
    else:
        name = "data/var_u"
        d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
        data = d['F']
        y = d['y']
        labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size)
        return (x_train, y_train), (x_test, y_test), 2





