import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from data.fabricated import get_fabricated
from data.InformationProcessor import InformationProcessor
from data.InfromationProcessorFabricatedData import InformationProcessorFabricatedData
from data.InformationProcessorUnion import InformationProcessorUnion

supported_data_sets = ["Tishby", "MNIST", "Fabricated"]


def get_information_processor(params):
    ds = params.data_set
    mi_e = params.mi_estimator
    ts = params.train_size
    fab = params.fabricated
    if ',' in params.mi_estimator:
        ips = [__get_information_processor(ds, mie, ts, filename(params, mie), fab) for mie in mi_e.split(',')]
        return InformationProcessorUnion(ips)

    return __get_information_processor(ds, mi_e, ts, filename(params), fab)


def __get_information_processor(data_set, mi_estimator, train_size, filename, fabricated=None):
    if data_set == 'MNIST':
        train, test, cat = get_mnist(train_size)
        return InformationProcessor(train, test, cat, filename, mi_estimator)
    elif data_set == "Tishby":
        train, test, cat = get_tishby(train_size)
        return InformationProcessor(train, test, cat, filename, mi_estimator)
    elif data_set == "Fabricated":
        train, test, cat, rel = get_fabricated(load_data(fabricated.base, train_size), fabricated.dim)
        return InformationProcessorFabricatedData(train, test, cat, rel, filename, mi_estimator)
    else:
        raise ValueError("Data set {} is not supported, supported data sets: {}"
                         .format(data_set, supported_data_sets))


def load_data(data_set, train_size, fabricated=None):
    if data_set == 'MNIST':
        return get_mnist(train_size)
    elif data_set == "Tishby":
        return get_tishby(train_size)
    elif data_set == "Fabricated":
        return get_fabricated(load_data(fabricated.base, train_size), fabricated.dim)
    else:
        raise ValueError("Data set {} is not supported, supported data sets: {}"
                         .format(data_set, supported_data_sets))


def get_mnist(train_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))

    data = data.reshape(data.shape[0], 28 * 28)
    data = data.astype('float32') / 255
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size)
    return (x_train, y_train), (x_test, y_test), 10


def get_tishby(train_size):
    name = "data/var_u"
    d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
    data = d['F']
    y = d['y']
    labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size)
    return (x_train, y_train), (x_test, y_test), 2


def filename(params, mie=None):
    name = "ts-" + "{0:.0%}".format(params.train_size) + ","
    if mie is None:
        name += "mie-" + str(params.mi_estimator) + ","
    else:
        name += "mie-" + str(mie) + ","
    name += "bs-" + str(params.batch_size) + ","
    name += "e-" + str(params.epochs) + ","
    name += "s-" + str(params.skip) + ","
    name += "ns-" + str(params.shape)
    name += "_" + params.data_set
    if params.data_set == 'Fabricated':
        name += "_d-" + str(params.fabricated.dim)
    return name
