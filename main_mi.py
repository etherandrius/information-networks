import information.information as inf
import networks.networks as networks
import _pickle
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from data.data import get_information_processor
from plot.plot import plot


def main():
    params = parameters()

    data_set = get_information_processor(params)
    (x_train, y_train), (x_test, y_test) = (data_set.x_train, data_set.y_train), (data_set.x_test, data_set.y_test)
    categories = data_set.categories
    model = networks.get_model_categorical(
        input_shape=x_train[0].shape, network_shape=params.shape, categories=categories)

    print("Training")
    information_callback = CalculateInformationCallback(
        model, data_set.information_calculator, x_test, params.skip, params.cores)
    model.fit(x_train, y_train,
              batch_size=params.batch_size,
              callbacks=[information_callback],
              epochs=params.epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    #fName = filename(params)
    print("Saving data to files")
    data_set.save()
    print("Producing image")
    data_set.plot()
    print("Done")
    return


print(__name__)
if __name__ == '__main__':
    print("Start")
    main()
