import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


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
