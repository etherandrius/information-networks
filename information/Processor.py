import numpy as np
import information.information as inf
import _pickle
from plot.plot import plot_main
from BlockingThreadPoolExecutor import BlockingThreadPoolExecutor
from threading import Lock


class InformationProcessor(object):
    def __init__(self, train, test, categories, filename=None, mi_estimator=None,
            delta=0.2, max_workers=4, bins=30):
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test
        self.categories = categories
        self.x_full = np.concatenate((self.x_train, self.x_test))
        self.y_full = np.concatenate((self.y_train, self.y_test))
        self.mi = {}
        self.__filename = filename
        self.__global_prev = None
        self.__buffered_activations = []
        self.__buffer_limit = 1
        self.__delta = delta
        self.__lock = Lock()
        self.__executor = BlockingThreadPoolExecutor(max_workers=max_workers)
        self.__calculator = inf.calculate_information(self.x_full, self.y_full, mi_estimator, bins)

    def save(self, append=""):
        path = "output/data/" + self.__filename + append + "_pickle"
        _pickle.dump(self.mi, open(path, 'wb'))

    def plot(self, append="", show=False):
        new_mi = list(zip(*map(lambda el: (el[0], *el[1]), self.mi.items())))
        epochs, i_x_t, i_y_t, i_t_t = new_mi
        path = "output/images/" + self.__filename + append
        plot_main(i_x_t, i_y_t, path, show)

    def calculate_information(self, activation, epoch):
        self.__lock.acquire()
        if self.__global_prev is None:
            self.__global_prev = self.__calculator(activation)
            self.mi[epoch] = self.__global_prev
            self.__lock.release()
            return

        self.__buffered_activations.append((activation, epoch))
        if len(self.__buffered_activations) >= self.__buffer_limit:
            # copy and clear __buffered_activations
            activation_buffer = self.__buffered_activations
            self.__buffered_activations = []

            local_prev = self.__global_prev

            # pre-compute next global_prev
            curr_activation, epoch_curr = activation_buffer[-1]

            mi_curr = self.__calculator(curr_activation)
            self.__global_prev = mi_curr
            if _dist(local_prev, mi_curr) <= self.__delta:
                self.__buffer_limit = min(self.__buffer_limit*2, 256)
            self.__lock.release()
            self.__executor.submit(self.__info_calc_entry, local_prev, mi_curr, epoch_curr, activation_buffer, [])
            return
        self.__lock.release()

    def finish_information_calculation(self):
        if len(self.__buffered_activations) > 0:
            self.__executor.submit(self.__info_calc_entry)
        self.__executor.shutdown()

    def __info_calc_entry(self, local_prev, mi_curr, epoch_curr, activation_buffer, carry):
        self._info_calc_inner_loop(local_prev, mi_curr, epoch_curr, activation_buffer, carry)
        with self.__lock:
            for epoch, mi in carry:
                self.mi[epoch] = mi

    def __info_calc_loop(self, mi_prev, activation_buffer, carry):
        assert(len(activation_buffer) > 0)
        curr_activation, epoch_curr = activation_buffer[-1]
        mi_curr = self.__calculator(curr_activation)

        return self._info_calc_inner_loop(mi_prev, mi_curr, epoch_curr, activation_buffer, carry)

    def _info_calc_inner_loop(self, mi_prev, mi_curr, epoch_curr, activation_buffer, carry):
        carry.append((epoch_curr, mi_curr))
        while _dist(mi_prev, mi_curr) > self.__delta:
            split = int(len(activation_buffer) / 2)
            if split == 0:
                break  # _dist(i, i+1) > delta, no further division is possible
            mi_prev = self.__info_calc_loop(mi_prev, activation_buffer[:split], carry)
            activation_buffer = activation_buffer[split:]
        return mi_curr


def _dist(i_a, i_b):
    d = max(
        max(abs(i_a[0] - i_b[0])),
        max(abs(i_a[1] - i_b[1])),
        max(abs(i_a[2] - i_b[2])),
    )
    return d
