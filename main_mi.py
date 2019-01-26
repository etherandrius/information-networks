import networks.networks as networks
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from data.data import get_information_processor


def main():
    params = parameters()

    data_set = get_information_processor(params)
    (x_train, y_train), (x_test, y_test) = (data_set.x_train, data_set.y_train), (data_set.x_test, data_set.y_test)
    categories = data_set.categories
    model = networks.get_model_categorical(
        input_shape=x_train[0].shape, network_shape=params.shape, categories=categories, activation=params.activation)

    print("Training")
    batch_size = params.batch_size if params.batch_size > 0 else len(x_train)
    no_of_batches = (len(x_train) / batch_size) * params.epochs
    information_callback = CalculateInformationCallback(
        model, data_set.information_calculator, data_set.x_full, params.skip, params.cores, no_of_batches)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[information_callback],
              epochs=params.epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    append = ",b-" + str(information_callback.batch)
    print("Saving data to files")
    data_set.save(append=append + "_pickle")
    print("Producing image")
    data_set.plot(append=append)
    print("Done")
    return


print(__name__)
if __name__ == '__main__':
    print("Start")
    main()
