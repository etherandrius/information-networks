import math
import keras
import networks.networks as networks
import numpy as np
from data.data import load_data
from keras import backend as K
from utils import bin_array, hash_data


def main():
    (x_train, y_train), (x_test, y_test), categories = load_data("Tishby", 0.1)

    model = networks.get_model_categorical(input_shape=x_train[0].shape, categories=categories)

    batch_size = 512
    epochs = 300
    no_of_batches = math.ceil(len(x_train) / batch_size) * epochs
    save_layers_callback = SaveLayers(model, x_test, max(no_of_batches - 10, 0))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[save_layers_callback],
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    print("Transforming data")
    saved = save_layers_callback.saved_layers

    saved_bin = [[bin_array(layer, bins=30, low=layer.min(), high=layer.max()) for layer in epoch] for epoch in saved]
    saved_hash = [[hash_data(layer) for layer in epoch] for epoch in saved_bin]

    t_data = {}
    for t in range(len(saved_hash[0])):
        t_data[t] = np.array([], dtype=np.int64)
        t_data[t] = saved_hash[0][t]
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            t_data[t] = np.concatenate([t_data[t], saved_hash[epoch][t]])

    tx_data = {}
    for t in range(len(saved_hash[0])):
        for xid in range(len(x_test)):
            tx_data[(t, xid)] = np.array([], dtype=np.int64)
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            for xid in range(len(saved_hash[0][0])):
                tx_data[(t, xid)] = np.append(tx_data[(t, xid)], saved_hash[epoch][t][xid])

    ty_data = {}
    for t in range(len(saved_hash[0])):
        for y in range(categories):
            ty_data[(t,y)] = np.array([], dtype=np.int64)
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            for xid, y in enumerate(y_test):
                y = np.where(y == 1)[0][0]
                ty_data[(t, y)] = np.append(ty_data[(t, y)], saved_hash[epoch][t][xid])

    print("Calculating probabilities")

    # Map: value -> probability

    py = get_probabilities([np.where(r == 1)[0][0] for r in y_test])
    px = get_probabilities(list(range(len(x_test))))  # x is just a uniform distribution
    pt = get_probabilities_map(t_data)
    ptx = get_probabilities_map(tx_data)
    pty = get_probabilities_map(ty_data)

    print("Calculating Mutual Information")

    # I(x, t) = sum(x,y){px*py*log(pxy/(px*py))}
    Ixt = {}
    for t in range(len(saved_hash[0])):
        Ixt[t] = 0
        for xid in range(len(x_test)):
            for vvv in ptx[(t, xid)]:
                pxpt = px[xid]*pt[t][vvv]
                pppp = ptx[(t, xid)][vvv]
                add = pppp*math.log2(pppp / pxpt)
                Ixt[t] += add

    print(Ixt)
    return


def get_probabilities(data):
    unique_array, unique_counts = np.unique(data, return_counts=True)
    prob = unique_counts / np.sum(unique_counts)
    return dict(zip(unique_array, prob))


def get_probabilities_map(map_data):
    prob = {}
    for key in map_data:
        prob[key] = get_probabilities(map_data[key])
    return prob


class SaveLayers(keras.callbacks.Callback):

    def __init__(self, model, x_test, save_layer=0):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        if type(save_layer) is int:
            self.__save_layer = lambda x: x > save_layer
        else:
            # assumes save_layer is a function
            self.__save_layer = save_layer

        # columns = ["x", "y"] + list(map(lambda x: "t" + str(x+1), list(range(len(model.layers)))))
        # self.saved_layers = pd.DataFrame(columns=columns)
        self.saved_layers = []
        self.__row_id = 0

        self.__batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test

    def on_batch_end(self, batch, logs=None):
        self.__batch += 1
        if self.__save_layer(self.__batch):
            out = self.__functor([self.__x_test, 0.])
            self.saved_layers.append(out)

    def get_saved_layers(self):
        return self.saved_layers


print(__name__)
if __name__ == '__main__':
    main()
