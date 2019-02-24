import numpy as np
import networks.networks as networks
from parameters import *
from information.CalculateInformationCallback import CalculateInformationCallback
from information.information import get_information_calculator
from data.data import load_data
from information.Processor import InformationProcessor
from information.ProcessorUnion import InformationProcessorUnion
import math
import keras as keras
from keras import backend as K
import pandas as pd
from tqdm import tqdm


def main():
    (x_train, y_train), (x_test, y_test), categories = load_data("Tishby", 0.8)

    model = networks.get_model_categorical(input_shape=x_train[0].shape, categories=categories)

    batch_size = 512
    epochs = 102
    no_of_batches = math.ceil(len(x_train) / batch_size) * epochs
    save_layers_callback = SaveLayers(model, x_test, y_test, no_of_batches - 4)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[save_layers_callback],
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)
    return


class SaveLayers(keras.callbacks.Callback):

    def __init__(self, model, x_test, y_test, save_layer=0):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        if type(save_layer) is int:
            self.__save_layer = lambda x: x > save_layer
        else:
            # assumes save_layer is a function
            self.__save_layer = save_layer

        columns = ["x", "y"] + list(map(lambda x: "t" + str(x+1), list(range(len(model.layers)))))
        self.saved_layers = pd.DataFrame(columns=columns)
        self.__row_id = 0

        self.__batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.__y_test = y_test

    def on_batch_end(self, batch, logs=None):
        self.__batch += 1
        print(self.__batch)
        if self.__save_layer(self.__batch):
            print(self.__batch)
            out = self.__functor([self.__x_test, 0.])
            for x_id, x in enumerate(self.__x_test):  # len(x_test) == len(y_test) == len(out[i])
                y = self.__y_test[x_id]
                row = [x, y]
                for layer in out:
                    row.append(layer[x_id])
                self.saved_layers.loc[self.__row_id] = row
                self.__row_id += 1

    def get_saved_layers(self):
        return self.saved_layers


print(__name__)
if __name__ == '__main__':
    main()
