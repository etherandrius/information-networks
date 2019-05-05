import keras
import numpy as np
from information.NaftaliTishby import __conditional_entropy, entropy_of_data
from keras import backend as K
from utils import bin_array, hash_data, pairwise
import _pickle
import argparse
import math
from networks.networks import network_parameters, get_model_categorical
from data.data import load_data, parameters_data
from plot.plot import plot_main


def main():
    args = get_parameters()

    filename(args)

    (x_train, y_train), (x_test, y_test), categories = load_data(args.data_set, args.train_size)
    no_of_batches = math.ceil(len(x_train) / args.batch_size) * args.epochs

    epoch_list = args.epoch_list
    if epoch_list[-1][1] > no_of_batches:
        raise ValueError("ranges out of range of training batches, number of batches {}, out of range value {}".format(no_of_batches, epoch_list[-1]))

    model = get_model_categorical(
        input_shape=x_train[0].shape,
        network_shape=args.shape,
        categories=categories,
        activation=args.activation)
    print("batches {}".format(no_of_batches))
    save_layers_callback = SaveLayers(model, x_test, epoch_list)
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              callbacks=[save_layers_callback],
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    def compute_single(saved, dist):
        x_test_hash = hash_data(x_test)
        data_x = x_test_hash
        for _ in range(dist - 1):
            data_x = np.concatenate((data_x, x_test_hash))

        y_test_hash = hash_data(y_test)
        data_y = y_test_hash
        for _ in range(dist - 1):
            data_y = np.concatenate((data_y, y_test_hash))

        # saved data where every number is binned
        saved_bin = [[bin_array(layer, bins=args.bins, low=layer.min(), high=layer.max()) for layer in epoch] for epoch in saved]
        # saved data where every number is hashed
        saved_hash = [[hash_data(layer) for layer in epoch] for epoch in saved_bin]

        data_t = {}
        for t in range(len(saved_hash[0])):
            data_t[t] = np.array([], dtype=np.int64)
        for epoch in range(len(saved_hash)):
            for t in range(len(saved_hash[0])):
                data_t[t] = np.concatenate([data_t[t], saved_hash[epoch][t]])
        data_t = list(data_t.values())

        h_t = np.array([entropy_of_data(t) for t in data_t])
        h_t_x = np.array([__conditional_entropy(t, data_x) for t in data_t])
        h_t_y = np.array([__conditional_entropy(t, data_y) for t in data_t])

        i_x_t = h_t - h_t_x
        i_y_t = h_t - h_t_y

        return i_x_t, i_y_t

    saved = save_layers_callback.saved_layers
    IXT, IYT = [], []
    pickle = {}
    for s, r in zip(saved, epoch_list):
        print("computing information for layers {}".format(r), end="")
        start, end = r
        dist = end - start
        ixt, iyt = compute_single(s, dist)
        print("  {} {}".format(ixt, iyt))
        pickle[start] = (ixt, iyt, [])
        IXT.append(ixt)
        IYT.append(iyt)

    path = args.dest + "/data/as_if_random/" + filename(args)
    _pickle.dump(pickle, open(path, 'wb'))
    path = args.dest + "/images/as_if_random/" + filename(args)
    plot_main(IXT, IYT, filename=path, show=True)

    return


def get_probabilities(data):
    unique_array, unique_counts = np.unique(data, return_counts=True)
    prob = unique_counts / np.sum(unique_counts)
    return dict(zip(unique_array, prob))


def get_probabilities_map(map_data):
    prob = {}
    for key in map_data:
        prob[key] = get_probabilities(map_data[key])
    return prob


class SaveLayers(keras.callbacks.Callback):

    def __init__(self, model, x_test, epoch_list):
        super().__init__()
        outputs = [layer.output for layer in model.layers]

        # columns = ["x", "y"] + list(map(lambda x: "t" + str(x+1), list(range(len(model.layers)))))
        # self.saved_layers = pd.DataFrame(columns=columns)
        self.epoch_list = epoch_list
        self.saved_layers = []
        self.__temp = []
        self.ix = 0
        self.__row_id = 0

        self.__batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test

    def in_range(self):
        ix = self.ix
        el = self.epoch_list
        if ix >= len(el):
            return False
        return (el[ix][0] <= self.__batch) and (self.__batch < el[ix][1])

    def on_batch_end(self, batch, logs=None):
        self.__batch += 1
        temp = self.__temp
        ix = self.ix
        el = self.epoch_list

        if ix >= len(self.epoch_list):
            return
        if (el[ix][0] <= self.__batch) and (self.__batch < el[ix][1]):
            out = self.__functor([self.__x_test, 0.])
            temp.append(out)
        if self.__batch >= el[ix][1]:
            self.ix += 1
            if len(temp) > 0:
                self.saved_layers.append(temp)
                self.__temp = []

    def get_saved_layers(self):
        return self.saved_layers


def get_parameters():
    parser = argparse.ArgumentParser()
    parameters_data(parser)
    network_parameters(parser)

    parser.add_argument('--bins', '-b',
                        dest='bins', default=30, type=int,
                        help="select the number of bins to use for bnning defaults to 30")

    parser.add_argument('--dest',
                        dest='dest', default="output",
                        help="destination folder for output files")

    parser.add_argument('--epoch_list', '-el',
                        dest='epoch_list', default="1-19,20-90",
                        help="list of ranges for which to compute mi. ex: 1-10,15-30, eanges assumed to be disjoint")

    parser.add_argument('--saved_epochs', '-se',
                        dest='no_saved_epochs', default=100, type=int,
                        help="no of epochs to consider when calculating mutual information")

    args = parser.parse_args()

    def convert_to_range(rrange):
        split = rrange.split('-')
        return int(split[0]), int(split[1])

    args.epoch_list = args.epoch_list
    args.epoch_list = args.epoch_list.split(',')
    args.epoch_list = [convert_to_range(e) for e in args.epoch_list]

    for s, e in args.epoch_list:
        if s > e:
            raise ValueError("invalid range {}".format((s, e)))
        if s == e:
            raise ValueError("range is 0, start {}, end {}".format(s, e))

    for a, b in pairwise(args.epoch_list):
        _, e = a
        s, _ = b
        if s < e:
            raise ValueError("ranges {} and {} overlap".format(a, b))

    return args


def filename(args):
    name = "ts-" + "{0:.0%}".format(args.train_size) + ","
    name += "e-" + str(args.epochs) + ","
    name += "_" + args.data_set + ","
    name += "_" + args.activation + ","
    name += "bs-" + str(args.batch_size) + ","
    name += "ns-" + str(args.shape)
    name += "el-" + str(args.epoch_list) + ","
    name += "_as_if_random"
    return name


print(__name__)
if __name__ == '__main__':
    main()
