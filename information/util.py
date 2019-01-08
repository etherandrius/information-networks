import numpy as np
import tensorflow.keras as keras
from threading import Lock
from tensorflow.keras import backend as K
from BlockingThreadPoolExecutor import BlockingThreadPoolExecutor


def binarize(data):
    if len(data.shape) < 2:
        return data
    return np \
        .ascontiguousarray(data) \
        .view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))


class CalculateInformationCallback(keras.callbacks.Callback):

    def __init__(self, model, calc_information_func, x_test, y_test, skip, max_workers):
        super().__init__()
        self.mi = []
        outputs = [layer.output for layer in model.layers]
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.__y_test = y_test
        self.__calc_information_func = calc_information_func
        self.__skip = skip
        self.__thread_executor = BlockingThreadPoolExecutor(max_workers=max_workers)
        self.__lock = Lock()
        self.batch = 0

    def __consume(self, out):
        mutual_information = self.__calc_information_func(out, self.__x_test, self.__y_test)
        self.__lock.acquire()
        self.mi.append(mutual_information)
        self.__lock.release()

    def on_batch_end(self, batch, logs=None):
        if self.batch % self.__skip == 0:
            out = self.__functor([self.__x_test, 0.])
            self.__thread_executor.submit(self.__consume, out)
        self.batch += 1

    def on_train_end(self, logs=None):
        print("Waiting on mutual information computations...")
        self.__thread_executor.shutdown()
        print("Mutual Information calculated")
