import networks.networks as networks
import information.information as information
import matplotlib.pyplot as plt
import time, numpy
import _pickle as cPickle
from tqdm import tqdm
from util import *
from information.util import CalculateInformationCallback


def main():
    args = parameters()
    train_size = args.train_size
    batch_size = args.batch_size
    epochs = args.epochs
    skip = args.skip
    shape = args.shape
    cores = args.cores
    data_set = args.data_set

    (x_train, y_train), (x_test, y_test), categories = load_data(data_set, train_size)
    model = networks.get_model_categorical(input_shape=x_train[0].shape, network_shape=shape, categories=categories)

    print("Training")
    information_callback = CalculateInformationCallback(
        model, information.calculate_information, x_test, y_test, skip, cores)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[information_callback],
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    i_x_t, i_y_t = zip(*information_callback.mi)

    filename = "train_size-" + "{0:.0%}".format(train_size) + ","
    filename += "batch_size-" + str(batch_size) + ","
    filename += "epochs-" + str(epochs) + ","
    filename += "mini_batches-" + str(information_callback.batch) + ","
    filename += "skip-" + str(skip) + ","
    filename += "shape-" + str(shape)
    if data_set == 'MNIST':
        filename += "_mnist"

    print("Saving data to file : ", filename)
    cPickle.dump(information_callback.mi, open("output/data/" + filename, 'wb'))

    print("Producing image")
    plot(i_x_t, i_y_t, show=False, filename="output/images/" + filename)

    print("Done")
    return


def plot(data_x, data_y, show=False, filename=None):
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    for ix in tqdm(range(len(data_x))):
    #for ix, (ex, ey) in tqdm(enumerate(zip(data_x, data_y))):
        ex = data_x[ix]
        ey = data_y[ix]
        for e in pairwise(zip(ex, ey)):
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


if __name__ == '__main__':
    main()
