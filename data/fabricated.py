# produces data set with relevant and irrelevant inputs, in order to better test mutual information;
import numpy as np


def get_fabricated(base_data, irr):
    (x_train, y_train), (x_test, y_test), categories = base_data
    rel = x_train[0].shape[0]

    x_train = np.array([np.array(np.append(x, random(irr))) for x in x_train])
    x_test = np.array([np.array(np.append(x, random(irr))) for x in x_test])

    return (x_train, y_train), (x_test, y_test), categories, rel


def random(dim):
    m = np.random.random_integers(0, dim)
    arr = np.ones(dim)
    arr[:m] = 0
    np.random.shuffle(arr)
    return arr



