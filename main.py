from util import load_data
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import networks.networks as networks
import information.information as information
import matplotlib.pyplot as plt
import itertools

class EveryEpoch(keras.callbacks.Callback):
    def __init__(self, model, x_test):
        self.activations = []
        outputs = [layer.output for layer in model.layers] + [model.output]
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test

    def on_train_begin(self, logs=None):
        self.activations = []  # epoch*layers*test_case*neuron -> value)

    def on_batch_end(self, batch, logs=None):
        self.activations.append(self.__functor([self.__x_test, 0.]))


def main():
    data = load_data()
    model = networks.get_model_categorical(input_shape=data.data[0].shape)
    # train = [train_the_model_get_distribution(1) for i in range(2)]  # itt*test_case*layer -> perceptron value
    train, test = data.split()
    print(len(train.data))

    layers_out = []

    x_test, y_test = test.data, test.labels

    x_train, y_train = train.data, train.labels
    every_epoch = EveryEpoch(model, test.data)
    model.fit(x_train, y_train,
              batch_size=512,
              callbacks=[every_epoch],
              epochs=1,
              validation_data=(x_test, y_test),
              verbose=1)

    activations = every_epoch.activations  # epoch*layers*test_case*neuron -> value)

    activations = activations[:1]

    i_x_t, i_y_t = zip(*[information.calculate_information(i, x_test, y_test) for i in activations])
    plot(i_x_t, i_y_t)
    return

def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)


def plot(data_x, data_y):
    for ex, ey in zip(data_x, data_y):
        for e in pairwise(zip(ex, ey)):
            print(e)
            (x1, y1), (x2, y2) = e
            plt.plot(x1, x2, y1, y2)
            #plt.plot(e)

    plt.scatter(data_x, data_y)
    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')
    plt.show()


if __name__ == '__main__':
    main()
