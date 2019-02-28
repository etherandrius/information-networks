import _pickle
from plot.plot import plot_main
from BlockingThreadPoolExecutor import BlockingThreadPoolExecutor
from threading import Lock


class InformationProcessorDeltaExact(object):
    """
    Contains the logic for which epochs to calculate Mutual Information
    """
    def __init__(self, information_calculator, buffer_limit=1, delta=0.2, max_workers=4):
        self.mi = {}
        self.__global_prev = None
        self.__buffered_activations = []
        self.__buffer_limit = buffer_limit
        self.__delta = delta
        self.__lock = Lock()
        self.__calculator = information_calculator
        self.__executor = BlockingThreadPoolExecutor(max_workers=max_workers)

    def save(self, path):
        _pickle.dump(self.mi, open(path, 'wb'))

    def plot(self, path, show=False):
        new_mi = list(zip(*map(lambda el: (el[0], *el[1]), self.mi.items())))
        epochs, i_x_t, i_y_t, i_t_t = new_mi
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
                self.__buffer_limit = min(self.__buffer_limit*2, 32)
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


def information_processor_parameters(parser):
    parameters = parser.add_argument_group('Information Processor parameters')

    parameters.add_argument('--delta',
                    '-d', dest='delta', default=0.1,
                    type=float, help="Tolerance on how densely to calculate mutual information, higher delta will skip more epochs")

    parameters.add_argument('--cores',
                            '-c', dest='cores', default=1,
                            type=int,
                            help='number of information instances to compute at a time')


def _dist(i_a, i_b):
    """
    Just a random distance metric used to decide if to compute mutual
    information for nearby epochs

    :param i_a: information for epoch a 
    :param i_b: information for epoch b
    :return: some notion of distance 
    """
    d = max(
        max(abs(i_a[0] - i_b[0])),
        max(abs(i_a[1] - i_b[1])),
        max(abs(i_a[2] - i_b[2])),
    )
    return d
