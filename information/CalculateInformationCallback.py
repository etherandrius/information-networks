import keras as keras
from keras import backend as K
from tqdm import tqdm


class CalculateInformationCallback(keras.callbacks.Callback):
    def __init__(self, model, information_processor, x_test, no_of_batches, progress_bar=True):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        self.batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.__ip = information_processor
        self.__progress_bar = progress_bar
        if progress_bar:
            self.__progress = tqdm(total=no_of_batches)

    def on_batch_end(self, batch, logs=None):
        if self.__progress_bar:
            self.__progress.update(1)
        self.batch += 1
        out = self.__functor([self.__x_test, 0.])
        self.__ip.calculate_information(out, self.batch)

    def on_train_end(self, logs=None):
        self.__ip.finish_information_calculation()
        if self.__progress_bar:
            self.__progress.close()

