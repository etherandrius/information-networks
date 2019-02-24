from keras.layers import Dropout, Dense, BatchNormalization
from keras import Sequential

activation_functions = ["tanh, sigmoid, relu, linear"]


def get_model_categorical(input_shape, network_shape="10,8,6,4", categories=2, activation='tanh'):
    model = Sequential()
    network_shape = network_shape.split(',')

    model.add(Dense(input_shape[0], activation=activation, input_shape=input_shape))

    for layer_spec in network_shape:
        model.add(decode_layer(layer_spec, activation))

    model.add(Dense(categories, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def decode_layer(spec, activation="tanh"):
    if spec == "Dropout" or spec == "Dr":
        spec = spec.split('-')
        dropout_rate = float(spec[1]) if len(spec) >= 2 else 0.2
        return Dropout(rate=dropout_rate)
    elif spec == "BatchNormalization" or spec == "BN":
        return BatchNormalization()
    else:
        spec = spec.split('-')
        size = int(spec[0])
        activation = spec[1] if len(spec) >= 2 else activation
        return Dense(size, activation=activation)
