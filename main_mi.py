import networks.networks as networks
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from data.data import get_information_processor
import math


def main():
    params = parameters()

    processor = get_information_processor(params)
    (x_train, y_train), (x_test, y_test) = (processor.x_train, processor.y_train), (processor.x_test, processor.y_test)
    categories = processor.categories
    model = networks.get_model_categorical(
        input_shape=x_train[0].shape, network_shape=params.shape, categories=categories, activation=params.activation)

    print("Training and Calculating mutual information")
    batch_size = min(params.batch_size, len(x_train)) if params.batch_size > 0 else len(x_train)
    no_of_batches = math.ceil(len(x_train) / batch_size) * params.epochs
    information_callback = CalculateInformationCallback(
        model, processor, processor.x_full, no_of_batches)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[information_callback],
              epochs=params.epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    append = ",b-" + str(information_callback.batch)
    print("Saving data to files")
    processor.save(append=append)
    print("Producing image")
    processor.plot(append=append)
    print("Done")
    return


print(__name__)
if __name__ == '__main__':
    print("Start")
    main()
