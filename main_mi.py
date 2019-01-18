import information.information as information
import networks.networks as networks
import matplotlib.pyplot as plt
import _pickle
import time
from tqdm import tqdm
from util import *
from information.CalculateInformationCallback import CalculateInformationCallback


def main():
    params = parameters()

    (x_train, y_train), (x_test, y_test), categories = load_data(params.data_set, params.train_size)
    model = networks.get_model_categorical(input_shape=x_train[0].shape, network_shape=params.shape, categories=categories)

    print("Training")
    information_callback = CalculateInformationCallback(
        model,
        information.calculate_information(x_test, y_test, params.mi_estimator), x_test, params.skip, params.cores)
    model.fit(x_train, y_train,
              batch_size=params.batch_size,
              callbacks=[information_callback],
              epochs=params.epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    i_x_t, i_y_t = zip(*information_callback.mi)
    fname = filename()
    print("Saving data to file : ", fname)
    _pickle.dump(information_callback.mi, open("output/data/" + fname, 'wb'))
    print("Producing image")
    plot(i_x_t, i_y_t, show=False, filename="output/images/" + fname)
    print("Done")
    return


def plot(data_x, data_y, show=False, filename=None):
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    for ix in tqdm(range(len(data_x))):
        for e in pairwise(zip(data_x[ix], data_y[ix])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)

    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()


def filename(params):
    name = "ts-" + "{0:.0%}".format(params.train_size) + ","
    name += "mie-" + str(params.mi_estimator) + ","
    name += "bs-" + str(params.batch_size) + ","
    name += "e-" + str(params.epochs) + ","
    name += "mini_batches-" + str(params.information_callback.batch) + ","
    name += "s-" + str(params.skip) + ","
    name += "ns-" + str(params.shape)
    if params.data_set == 'MNIST':
        name += "_mnist"
    return name


print(__name__)
if __name__ == '__main__':
    print("Start")
    main()
