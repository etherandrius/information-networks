import itertools
import numpy as np


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)


# addding noise is necessary to prevent infinite MI (i.e prevents division by zero for some MI estimators)
def add_noise(data, mean=0, std=0.01):
    return data + np.random.normal(mean, std, data.shape)


def __add_noise_value(n, noise_function):
    return n + noise_function()
