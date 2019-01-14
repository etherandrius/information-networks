import numpy as np
from information.util import binarize, add_noise
import lnn.lnn as lnn


def get_probabilities(data):
    unique_array, _, unique_inverse, unique_counts = \
        np.unique(data, return_counts=True, return_index=True, return_inverse=True)
    prob = unique_counts / np.sum(unique_counts)

    return prob, unique_inverse


# low=inclusive, high=exclusive
def bin_value(n, low=-1, high=1, batches=10):
    e = (high - low) / batches
    return max(min(int((n - low) / e), batches - 1), 0)


def bin_array(N, low=-1, high=1, batches=10):
    result = []
    for n in N:
        if isinstance(n, np.ndarray) or isinstance(n, list):
            new_n = bin_array(n, low, high, batches)
        else:
            new_n = bin_value(n, low, high, batches)
        result.append(new_n)
    return result


def entropy_of_probabilities(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))


def entropy_of_data(data):
    prob_data, _ = get_probabilities(data)
    return entropy_of_probabilities(prob_data)


def conditional_entropy(data_y, data_x):
    # H(Y|X)
    # assumes every x is unique
    p_x, x_to_y = get_probabilities(data_x)
    entropy = 0
    for ix, px in enumerate(p_x):
        h_y_given_x = entropy_of_data(data_y[x_to_y == ix])
        entropy += px * h_y_given_x

    return entropy


def calculate_information_data(data_x, data_y):
    x = binarize(data_x)
    y = binarize(data_y)

    h_x = entropy_of_data(x)
    h_x_y = conditional_entropy(x, y)

    mutual_information = h_x - h_x_y
    return mutual_information


def calculate_information_lnn(input_values, labels, entropy):
    data_x = input_values
    data_y = labels

    def noise():
        np.random.normal(0, 0.7, 1)[0] # 0.7 ~= sqrt(0.5)
    data_x = add_noise(input_values, noise)
    data_y = add_noise(labels, noise)

    if entropy == "KL":
        entropy = lnn.KL_entropy
    elif entropy == "KDE":
        entropy = lnn.KDE_entropy
    elif entropy == "LNN_2":
        entropy = lnn.LNN_2_entropy
    elif entropy == "LNN_1":
        entropy = lnn.LNN_1_entropy

    e_y = entropy(data_y)
    e_x = entropy(data_x)

    def information(activation):
        # data_t = [add_noise(a, noise) for a in activation]
        data_t = activation # don't think need to add noise to activations as they are produced randomly by the neural
        # network training algorithm

        e_t_y = [entropy(np.array([np.append(t, y) for t, y in zip(layer, data_y)])) for layer in data_t]
        e_t_x = [entropy(np.array([np.append(t, x) for t, x in zip(layer, data_x)])) for layer in data_t]
        i_x_t = e_x + e_y - e_t_x
        i_y_t = e_x + e_y - e_t_y

        return i_x_t, i_y_t

    return information


def calculate_information(input_values, labels):
    # activation layers*test_case*neuron -> value)

    # calculate information I(X,T) and I(T,Y) where X is the input and Y is the output
    # and T is any layer
    def information(activation):
        data_t = activation

        data_t = [np.asarray(bin_array(t)) for t in data_t]

        data_x = binarize(input_values)
        data_y = binarize(labels)
        data_t = [binarize(t) for t in data_t]

        h_t = np.array([entropy_of_data(t) for t in data_t])
        h_t_x = np.array([conditional_entropy(t, data_x) for t in data_t])
        h_t_y = np.array([conditional_entropy(t, data_y) for t in data_t])

        i_x_t = h_t - h_t_x
        i_y_t = h_t - h_t_y

        return i_x_t, i_y_t

    return information
