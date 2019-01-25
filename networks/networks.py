import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

activation_functions = ["tanh, sigmoid, relu, linear"]


def get_model_categorical(input_shape, network_shape, categories=2, activation='tanh'):
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignores warning caused by callbacks being expensive
    model = Sequential()
    network_shape = network_shape.split(',')
    size, layer, func = decode_layer_spec(network_shape[0], activation)
    model.add(layer(size, activation=func, input_shape=input_shape))

    for layer_spec in network_shape[1:-1]:
        model.add(decode_layer(layer_spec, activation))

    size, layer, func = decode_layer_spec(network_shape[0], activation)
    model.add(layer(size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def decode_layer(spec, activation="tanh"):
    size, layer, func = decode_layer_spec(spec, activation)
    if layer == "Dense" or layer == "D":
        return Dense(size, activation=func)
    elif layer == "Dropout" or layer == "Dr":
        return Dropout()
    elif layer == "BatchNormalization" or layer == "BN":
        return BatchNormalization()


def decode_layer_spec(spec, activation="tanh"):
    spec = spec.split('-')
    size = int(spec[0])
    layer = spec[1] if len(spec) > 1 else "Dense"
    func = spec[2] if len(spec) > 2 else activation
    return size, layer, func


# trains the model and records value of the activated percpetron for every layer for every test element in x_test,
# so returns an array of [test_case][layer][perceptron value]
def train_the_model_get_distribution(random=42):
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(16, activation='tanh', input_shape=x_train[0].shape))
    model.add(Dense(14, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test), verbose=1)

    model.out

    outputs = [model.input] + [layer.output for layer in model.layers] + [model.output]
    functor = K.function([model.input, K.learning_phase()], outputs)

    layer_outs = functor([x_test, 0.])
    return layer_outs
