import itertools
import numpy as np


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)


def noise(mean=0, std=0.01):
    return np.random.normal(mean, std, 1)[0]


# addding noise is necessary to prevent infinite MI (i.e prevents division by zero for some MI estimators)
def add_noise(data, noise_function):
    result = []
    for n in data:
        if isinstance(n, np.ndarray) or isinstance(n, list):
            new_n = add_noise(n, noise_function)
        else:
            new_n = __add_noise_value(n, noise_function)
        result.append(new_n)
    return np.array(result)


def __add_noise_value(n, noise_function):
    return n + noise_function()
