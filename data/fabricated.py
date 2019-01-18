# produces data set with relevant and irrelevant inputs, in order to better test mutual information;
import numpy as np


def get_fabricated(base_data, dim):
    (x_train, y_train), (x_test, y_test), categories = base_data

    x_train = np.array([np.array(np.append(x, random(dim))) for x in x_train])
    x_test = np.array([np.array(np.append(x, random(dim))) for x in x_test])

    return base_data


def random(dim):
    m = np.random.random_integers(0, dim)
    arr = np.ones(dim)
    arr[:m] = 0
    np.random.shuffle(arr)
    return arr



