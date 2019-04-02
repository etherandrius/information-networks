import math
import keras
import networks.networks as networks
import numpy as np
from information.NaftaliTishby import __conditional_entropy, entropy_of_data
from data.data import load_data
from keras import backend as K
from utils import bin_array, hash_data


def main():

    A = np.array([1, 2, 1, 2, 1, 2])
    B = np.array([2, 2, 3, 3, 5, 5])
    C = np.array([1, 2, 3, 1, 2, 3])

    print("entropy")
    print(entropy_of_data(A))
    print(entropy_of_data(B))
    print(entropy_of_data(C))

    print("conditional entropy")
    print("A, B ", __conditional_entropy(A, B))
    print("A, C ", __conditional_entropy(A, C))
    print("B, A ", __conditional_entropy(B, A))
    print("B, C ", __conditional_entropy(B, C))
    print("C, A ", __conditional_entropy(C, A))
    print("C, B ", __conditional_entropy(C, B))

    (x_train, y_train), (x_test, y_test), categories = load_data("Tishby", 0.2)

    model = networks.get_model_categorical(input_shape=x_train[0].shape, categories=categories)

    batch_size = 512
    epochs = 1000
    no_saved_layers = 10
    no_of_batches = math.ceil(len(x_train) / batch_size) * epochs
    save_layers_callback = SaveLayers(model, x_test, max(no_of_batches - no_saved_layers, 0))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[save_layers_callback],
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    print("Transforming data")
    saved = save_layers_callback.saved_layers

    x_test_hash = hash_data(x_test)

    x_test_all = x_test_hash
    for _ in range(no_saved_layers-1):
        x_test_all = np.concatenate((x_test_all, x_test_hash))

    # saved data where every number is binned
    saved_bin = [[bin_array(layer, bins=30, low=layer.min(), high=layer.max()) for layer in epoch] for epoch in saved]
    # saved data where every number is hashed
    saved_hash = [[hash_data(layer) for layer in epoch] for epoch in saved_bin]

    t_data = {}
    for t in range(len(saved_hash[0])):
        t_data[t] = np.array([], dtype=np.int64)
        # feel like the line below should not be there I've commented it out
        # t_data[t] = saved_hash[0][t]
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            t_data[t] = np.concatenate([t_data[t], saved_hash[epoch][t]])

    tx_data = {}
    for t in range(len(saved_hash[0])):
        for x_id in range(len(x_test)):
            tx_data[(t, x_id)] = np.array([], dtype=np.int64)
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            for x_id in range(len(saved_hash[0][0])):
                tx_data[(t, x_id)] = np.append(tx_data[(t, x_id)], saved_hash[epoch][t][x_id])

    ty_data = {}
    for t in range(len(saved_hash[0])):
        for y in range(categories):
            ty_data[(t,y)] = np.array([], dtype=np.int64)
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            for x_id, y in enumerate(y_test):
                y = np.where(y == 1)[0][0]
                ty_data[(t, y)] = np.append(ty_data[(t, y)], saved_hash[epoch][t][x_id])

    print("Calculating probabilities")

    # Map: value -> probability

    # probabilities of p(y) for all y
    P_y = get_probabilities([np.where(r == 1)[0][0] for r in y_test])
    # probabilities of p(x) for all x
    P_x = get_probabilities(list(range(len(x_test))))  # x is just a uniform distribution
    # probabilities of p(t) for all t
    P_t = get_probabilities_map(t_data)
    # probabilities of p(t,x) for all t and x
    P_tx = get_probabilities_map(tx_data)
    # probabilities of p(t,y) for all t and y
    P_ty = get_probabilities_map(ty_data)

    print("Calculating Mutual Information")

    # I(x, y) = sum_over(x,y){p(x)*p(y)*log(p(x,y)/(p(x)*p(y)))}
    Ixt = {}
    for t in range(len(saved_hash[0])):
        Ixt[t] = 0
        for x_id in range(len(x_test)):
            for tx_id in P_tx[(t, x_id)]:
                p_x = P_x[x_id]
                p_t = P_t[t][tx_id]
                p_xt = P_tx[(t, x_id)][tx_id]
                add = p_x*p_t*math.log2(p_xt / p_x*p_t)
                Ixt[t] += add

    Iyt = {}
    for t in range(len(saved_hash[0])):
        Iyt[t] = 0
        for y_id in range(categories):
            for ty_id in P_ty[(t, y_id)]:
                p_y = P_y[y_id]
                p_t = P_t[t][ty_id]
                p_yt = P_ty[(t, y_id)][ty_id]
                add = p_y*p_t*math.log2(p_yt / p_y*p_t)
                Iyt[t] += add

    print(Ixt)
    print(Iyt)
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
