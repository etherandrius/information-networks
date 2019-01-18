import information.information as information
import networks.networks as networks
import _pickle
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from data.data import load_data
from plot.plot import plot


def main():
    params = parameters()

    (x_train, y_train), (x_test, y_test), categories = load_data(params.data_set, params.train_size)
    model = networks.get_model_categorical(
        input_shape=x_train[0].shape, network_shape=params.shape, categories=categories)

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
    fName = filename()
    print("Saving data to file : ", fName)
    _pickle.dump(information_callback.mi, open("output/data/" + fName, 'wb'))
    print("Producing image")
    plot(i_x_t, i_y_t, show=False, filename="output/images/" + fName)
    print("Done")
    return


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
