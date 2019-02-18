import numpy as np
from utils import pairwise


def binarize(data):
    if len(data.shape) < 2:
        return data
    return np \
        .ascontiguousarray(data) \
        .view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))


def get_probabilities(data):
    unique_array, _, unique_inverse, unique_counts = \
        np.unique(data, return_counts=True, return_index=True, return_inverse=True)
    prob = unique_counts / np.sum(unique_counts)

    return prob, unique_inverse


# low=inclusive, high=exclusive
def bin_value(n, low=-1, high=1, bins=10):
    e = (high - low) / bins
    return max(min(int((n - low) / e), bins - 1), 0)


def bin_array(array, low=-1, high=1, bins=10):
    result = []
    for n in array:
        if isinstance(n, np.ndarray) or isinstance(n, list):
            new_n = bin_array(n, low, high, bins)
        else:
            new_n = bin_value(n, low, high, bins)
        result.append(new_n)
    return result


def entropy_of_probabilities(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))


def entropy_of_data(data):
    prob_data, _ = get_probabilities(data)
    e = entropy_of_probabilities(prob_data)
    return e


def bin_then_entropy(data):
    bined_data = np.asarray(bin_array(data))
    binar_data = binarize(bined_data)
    e = entropy_of_data(binar_data)
    return e


def __conditional_entropy(data_y, data_x):
    # H(Y|X)
    # assumes every x is unique
    p_x, x_to_y = get_probabilities(data_x)
    entropy = 0
    for ix, px in enumerate(p_x):
        h_y_given_x = entropy_of_data(data_y[x_to_y == ix])
        entropy += px * h_y_given_x

    return entropy


def __calculate_information_data(data_x, data_y):
    x = binarize(data_x)
    y = binarize(data_y)

    h_x = entropy_of_data(x)
    h_x_y = __conditional_entropy(x, y)

    mutual_information = h_x - h_x_y
    return mutual_information


def __calculate_information_tishby(input_values, labels, bins=30):

    # activation layers*test_case*neuron -> value)

    # calculate information I(X,T) and I(T,Y) where X is the input and Y is the output
    # and T is any layer
    data_x = binarize(input_values)
    data_y = binarize(labels)

    def information(activation):
        data_t = activation

        if bins == -1:
            data_t = [np.asarray(bin_array(t, bins=30, low=t.min(), high=t.max())) for t in data_t]
        else:
            data_t = [np.asarray(bin_array(t, bins=bins, low=t.min(), high=t.max())) for t in data_t]

        data_t = [binarize(t) for t in data_t]

        h_t = np.array([entropy_of_data(t) for t in data_t])
        h_t_x = np.array([__conditional_entropy(t, data_x) for t in data_t])
        h_t_y = np.array([__conditional_entropy(t, data_y) for t in data_t])

        h_t_t = np.array([__conditional_entropy(t1, t2) for (t1, t2) in pairwise(data_t)])

        i_x_t = h_t - h_t_x
        i_y_t = h_t - h_t_y
        i_t_t = h_t[:-1] - h_t_t

        return i_x_t, i_y_t, i_t_t

    return information

