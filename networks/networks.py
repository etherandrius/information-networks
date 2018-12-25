import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def get_model_categorical(input_shape, layer_sizes=[10, 7, 5, 4, 3], categories=2, activation='tanh'):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation=activation, input_shape=input_shape))
    for lsize in layer_sizes[1:]:
        model.add(Dense(lsize, activation=activation))
    model.add(Dense(categories, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


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