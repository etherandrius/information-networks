import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
import argparse
import itertools
import multiprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


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


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set',
                        '-ds', dest='data_set', default='default',
                        help='choose a data set, available: [default, MNIST], default - data set used by Tishby in the original paper')

    parser.add_argument('--train_size',
                        '-ts', dest='train_size', default=0.8,
                        type=float, help='Training size')

    parser.add_argument('--batch_size',
                        '-bs', dest='batch_size', default=512,
                        type=int)

    parser.add_argument('--information_batch_number',
                        '-bn', dest='batch_number', default=30,
                        type=int, help='Number of batches to be used for information calculation')

    parser.add_argument('--num_of_epochs',
                        '-e', dest='epochs', default=1500,
                        type=int, help='Number of times to scan the dataset for NN training')

    parser.add_argument('--skip',
                        '-s', dest='skip', default=1,
                        type=int, help="Calculate information for every n'th mini-batch epoch")

    parser.add_argument('--network_shape',
                        '-ns', dest='shape', default="12,10,8,6,4,2,1",
                        help='Shape of the DNN')

    parser.add_argument('--cores',
                        '-c', dest='cores', default=multiprocessing.cpu_count(),
                        type=int, help='How many cores to use for mutual information computation defaults to number of cores on the machine')

    parser.add_argument('--mi_estimator',
                        '-mie', dest='mi_estimator', default="bins",
                        help="Choose what mutual information estimator to use available: [bins, KDE, KL, LNN_1, LNN_2], "
                             "bins - method used by Tishby in his paper, "
                             "KDE - Kernel density estimator, "
                             "KL - Kozachenko-Leonenko estimator, "
                             "LNN_1, LNN_2 - Local nearest neighbour with order 1 or 2")

    args = parser.parse_args()
    args.shape = list(map(int, args.shape.split(',')))
    return args


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


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)
