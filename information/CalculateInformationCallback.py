import tensorflow.keras as keras
from threading import Lock
from tensorflow.keras import backend as K
from BlockingThreadPoolExecutor import BlockingThreadPoolExecutor
from tqdm import tqdm


class CalculateInformationCallback(keras.callbacks.Callback):

    def __init__(self, model, calc_information_func, x_test, distance, max_workers, no_of_batches):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.__calc_information_func = calc_information_func
        self.__skip = skip
        self.__thread_executor = BlockingThreadPoolExecutor(max_workers=max_workers)
        self.__lock = Lock()
        self.progress = tqdm(total=no_of_batches)

        self.batch = 0

    def consume(self, out):
        self.__calc_information_func(out)
        self.__lock.acquire()
        self.progress.update(1)
        self.__lock.release()

    def on_batch_end(self, batch, logs=None):
        self.progress.update(1)
        if self.batch % self.__skip == 0:
            out = self.__functor([self.__x_test, 0.])
            self.__thread_executor.submit(self.consume, out)
        self.batch += 1

    def on_train_end(self, logs=None):
        self.__thread_executor.shutdown()
        self.progress.close()

    def _dist(self, ):

