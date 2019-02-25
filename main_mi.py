import numpy as np
import networks.networks as networks
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from information.information import get_information_calculator
from data.data import load_data
from information.Processor import InformationProcessor
from information.ProcessorUnion import InformationProcessorUnion
import math


def main():
    params = general_parameters()

    (x_train, y_train), (x_test, y_test), categories = load_data(params.data_set, params.train_size)

    x_full = np.concatenate((x_train, x_test))
    y_full = np.concatenate((y_train, y_test))

    if ',' not in params.mi_estimator:
        information_calculator = get_information_calculator(x_full, y_full, params.mi_estimator, params.bins)
        processor = InformationProcessor(information_calculator)
    else:
        mies = params.mi_estimator.split(',')
        calculators = [get_information_calculator(x_full, y_full, mie, params.bins) for mie in mies]
        ips = [InformationProcessor(calc) for calc in calculators]
        processor = InformationProcessorUnion(ips)

    model = networks.get_model_categorical(
        input_shape=x_train[0].shape, network_shape=params.shape, categories=categories, activation=params.activation)

    print("Training and Calculating mutual information")
    batch_size = min(params.batch_size, len(x_train)) if params.batch_size > 0 else len(x_train)
    no_of_batches = math.ceil(len(x_train) / batch_size) * params.epochs
    information_callback = CalculateInformationCallback(
        model, processor, x_full, no_of_batches)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[information_callback],
              epochs=params.epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    append = ",b-" + str(information_callback.batch)
    print("Saving data to files")
    processor.save("output/data/" + filename(params) + append)
    print("Producing image")
    processor.plot("output/images/" + filename(params) + append)
    print("Done")
    return


def params():
    parser = argparse.ArgumentParser()
    networks.parameters_network(parser)
    general_parameters(parser)


def filename(params):
    name = "ts-" + "{0:.0%}".format(params.train_size) + ","
    name += "e-" + str(params.epochs) + ","
    name += "_" + params.activation
    name += "_" + params.data_set + ","
    name += "mie-" + str(params.mi_estimator) + ","
    name += "bs-" + str(params.batch_size) + ","
    if params.bins != 1:
        name += "bins-" + str(params.bins) + ","
    name += "ns-" + str(params.shape)
    return name


print(__name__)
if __name__ == '__main__':
    print("Start")
    main()
