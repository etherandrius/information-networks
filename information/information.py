import numpy as np
import information.WeihaoGao as wGao
import information.NaftaliTishby as nTishby

supported_estimators = ["KL", "KSG", "KDE", "LNN_1", "LNN_2", "bins"]


def calculate_information(input_values, labels, entropy):
    if entropy.startswith("bins"):
        return nTishby.__calculate_information_binning(input_values, labels, entropy)
    elif entropy == "KSG":
        return __calculate_information_KSG(input_values, labels)
    elif entropy == "bins2":
        entropy = nTishby.bin_then_entropy
    elif entropy == "KL":
        entropy = wGao.KL_entropy
    elif entropy == "KDE":
        entropy = wGao.KDE_entropy
    elif entropy == "LNN_2":
        entropy = wGao.LNN_2_entropy
    elif entropy == "LNN_1":
        entropy = wGao.LNN_1_entropy
    elif entropy is None:
        return lambda x: None
    else:
        raise ValueError("Unsuported mutual information estimator {}, available: {}".format(entropy, input_values))

    return __calculate_information_wgao(input_values, labels, entropy)


def __noise(mean=0, std=0.01):
    return np.random.normal(mean, std, 1)[0]


def __calculate_information_KSG(input_values, labels):
    data_x = input_values
    data_y = labels

    data_x = add_noise(input_values, __noise)
    data_y = add_noise(labels, __noise)

    def information(activation):
        data_t = activation
        #e_t = [entropy(t) for t in data_t]

        i_y_t = [wGao._KSG_mi(np.array([np.append(t, y) for t, y in zip(layer, data_y)]), split=len(layer[0])) for layer in data_t]
        i_x_t = [wGao._KSG_mi(np.array([np.append(t, x) for t, x in zip(layer, data_x)]), split=len(layer[0])) for layer in data_t]
        #i_t_x = [entropy(np.array([np.append(t, x) for t, x in zip(layer, data_x)])) for layer in data_t]
        #i_x_t = e_x + e_t - e_t_x
        #i_y_t = e_y + e_t - e_t_y

        return i_x_t, i_y_t

    return information


def __calculate_information_wgao(input_values, labels, entropy):
    data_x = input_values
    data_y = labels

    data_x = add_noise(input_values, __noise)
    data_y = add_noise(labels, __noise)

    e_y = entropy(data_y)
    e_x = entropy(data_x)

    def information(activation):
        # data_t = [add_noise(a, noise) for a in activation]
        data_t = activation  # don't think need to add noise to activations as they are produced randomly by the neural
        # network training algorithm, adding noise only prevents entropy calculations from failing in situations when 5
        # points have the exact same values, then a division by zero is possible.

        e_t = [entropy(t) for t in data_t]

        e_t_y = [entropy(np.array([np.append(t, y) for t, y in zip(layer, data_y)])) for layer in data_t]
        e_t_x = [entropy(np.array([np.append(t, x) for t, x in zip(layer, data_x)])) for layer in data_t]
        i_x_t = e_x + e_t - e_t_x
        i_y_t = e_y + e_t - e_t_y

        return i_x_t, i_y_t

    return information


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
