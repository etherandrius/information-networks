import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def main():
    train = [train_the_model_get_weights(1) for i in range(10)]


def train_the_model_get_weights(random=42):
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=x_train[0].shape))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test), verbose=1)

    outputs = [layer.output for layer in model.layers]
    functor = K.function([model.input, K.learning_phase()], outputs)

    layer_outs = functor([x_test, 0.])
    return layer_outs


if __name__ == '__main__':
    main()
