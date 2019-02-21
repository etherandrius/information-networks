import numpy as np
import information.WeihaoGao as wGao
import information.NaftaliTishby as nTishby
import information.kde as kde
from utils import pairwise, add_noise

supported_estimators = ["Tishby", "KDE"]


def get_information_calculator(input_values, labels, entropy, bins):
    if entropy is None or entropy == "None":
        return lambda x: None
    elif entropy == "bins" or entropy == "Tishby":
        return nTishby.__calculate_information_tishby(input_values, labels, bins)
    elif entropy == "KDE":
        return kde.calculate_information_saxe(input_values, labels, bins)
    elif entropy == "KSG":
        return __calculate_information_KSG(input_values, labels. bins)
    elif entropy == "bins2":
        entropy = nTishby.bin_then_entropy
    elif entropy == "KL":
        entropy = wGao.KL_entropy
    elif entropy == "bad-KDE":
        entropy = wGao.KDE_entropy
    elif entropy == "LNN_2":
        entropy = wGao.LNN_2_entropy
    elif entropy == "LNN_1":
        entropy = wGao.LNN_1_entropy
    else:
        raise ValueError("Unsupported mutual information estimator {}, available: {}".format(entropy, input_values))

    return __calculate_information_wgao(input_values, labels, entropy, bins)


def __calculate_information_KSG(input_values, labels, bins=30):
    data_x = input_values
    data_y = labels

    data_x = add_noise(input_values)
    data_y = add_noise(labels)

    def information(activation):
        data_t = activation
        #e_t = [entropy(t) for t in data_t]

        if bins > 0:
            data_t = [add_noise(np.asarray(nTishby.bin_array(t, bins=bins, low=t.min(), high=t.max()))) for t in data_t]

        i_y_t = [wGao._KSG_mi(np.array([np.append(t, y) for t, y in zip(layer, data_y)]), split=len(layer[0])) for layer in data_t]
        i_x_t = [wGao._KSG_mi(np.array([np.append(t, x) for t, x in zip(layer, data_x)]), split=len(layer[0])) for layer in data_t]
        i_t_t = [wGao._KSG_mi(np.array([np.append(a, b) for a, b in zip(t0, t1)]), split=len(t0[0])) for t0, t1 in pairwise(data_t)]

        #i_t_x = [entropy(np.array([np.append(t, x) for t, x in zip(layer, data_x)])) for layer in data_t]
        #i_x_t = e_x + e_t - e_t_x
        #i_y_t = e_y + e_t - e_t_y

        return i_x_t, i_y_t, i_t_t

    return information


def __calculate_information_wgao(input_values, labels, entropy, bins=30):
    data_x = input_values
    data_y = labels

    data_x = add_noise(input_values)
    data_y = add_noise(labels)

    e_y = entropy(data_y)
    e_x = entropy(data_x)

    def information(activation):
        # data_t = [add_noise(a, noise) for a in activation]
        data_t = activation  # don't think need to add noise to activations as they are produced randomly by the neural
        # network training algorithm, adding noise only prevents entropy calculations from failing in situations when 5
        # points have the exact same values, then a division by zero is possible.

        if bins > 0:
            data_t = [add_noise(np.asarray(nTishby.bin_array(t, bins=bins, low=t.min(), high=t.max()))) for t in data_t]

        e_t = np.array([entropy(t) for t in data_t])
        e_t_y = [entropy(np.array([np.append(t, y) for t, y in zip(layer, data_y)])) for layer in data_t]
        e_t_x = [entropy(np.array([np.append(t, x) for t, x in zip(layer, data_x)])) for layer in data_t]
        e_t_t = np.array([entropy(np.array([np.append(a, b) for a, b in zip(t1, t2)])) for t1, t2 in pairwise(data_t)])

        i_x_t = e_x + e_t - e_t_x
        i_y_t = e_y + e_t - e_t_y
        i_t_t = e_t[:-1] + e_t[1:] - e_t_t  # I(t0, t1) = H(t0) + H(t1) + H(t0, t1)

        return i_x_t, i_y_t, i_t_t

    return information
