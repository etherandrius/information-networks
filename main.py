from util import load_data
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from joblib import Parallel, delayed
import networks.networks as networks
import information.information as information
import matplotlib.pyplot as plt
from matplotlib import colorbar
import numpy as np
import itertools
import argparse
import multiprocessing
from tqdm import tqdm


class EveryEpoch(keras.callbacks.Callback):
    def __init__(self, model, x_test):
        self.activations = []
        #outputs = [layer.output for layer in model.layers] + [model.output]
        outputs = [layer.output for layer in model.layers]
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.batch = 0

    def on_train_begin(self, logs=None):
        self.activations = []  # epoch*layers*test_case*neuron -> value)

    def on_batch_end(self, batch, logs=None):
        self.activations.append(self.__functor([self.__x_test, 0.]))
        self.batch += 1


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size',
                        '-ts', dest='train_size', default=0.8,
                        type=float, help='Training size')

    parser.add_argument('--batch_size',
                        '-bs', dest='batch_size', default=512,
                        type=int)

    parser.add_argument('--information_batch_number',
                        '-bn', dest='batch_number', default=30,
                        type=int, help='Number of batches to be used for information calculation')
    parser.add_argument('--num_of_epochs',
                        '-e', dest='epochs', default=1500,
                        type=int, help='Number of times to scan the dataset for NN training')
    parser.add_argument('--skip',
                        '-s', dest='skip', default=1,
                        type=int, help="Calculate information for every n'th mini-batch epoch")
    parser.add_argument('--network_shape',
                        '-ns', dest='shape', default="12,10,8,6,4,2,1",
                        help='Shape of the DNN')

    args = parser.parse_args()
    args.shape = list(map(int, args.shape.split(',')))
    return args


def main():

    args = parameters()
    train_size = args.train_size
    batch_size = args.batch_size
    epochs = args.epochs
    skip = args.skip
    shape = args.shape

    data = load_data()
    model = networks.get_model_categorical(input_shape=data.data[0].shape, network_shape=shape)
    train, test = data.split(train_size)

    x_test, y_test = test.data, test.labels


    print("Training")
    x_train, y_train = train.data, train.labels
    every_epoch = EveryEpoch(model, test.data)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[every_epoch],
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    activations = every_epoch.activations  # epoch*layers*test_case*neuron -> value)
    activations = activations[::skip]

    print("Batches : ", every_epoch.batch)

    print("Calculating Information")
    i_x_t, i_y_t = zip(*
        #Parallel(n_jobs=multiprocessing.cpu_count())
        Parallel(n_jobs=2)
            (delayed(information.calculate_information)(i, x_test, y_test) for i in tqdm(activations)))

    print("Producing image")
    filename = "output/"
    filename += "train_size-" + "{0:.0%}".format(train_size) + ","
    filename += "batch_size-" + str(batch_size) + ","
    filename += "epochs-" + str(epochs) + ","
    filename += "mini_batches-" + str(every_epoch.batch) + ","
    filename += "skip-" + str(skip) + ","
    filename += "shape-" + str(shape)
    plot(i_x_t, i_y_t, show=False, filename=filename)
    return


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)


def plot(data_x, data_y, show=False, filename=None):
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    for ix, (ex, ey) in enumerate(zip(data_x, data_y)):
        for e in pairwise(zip(ex, ey)):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)

    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    if filename is not None:
        plt.savefig(filename, dpi=1000)
    if show:
        plt.show()


if __name__ == '__main__':
    main()
