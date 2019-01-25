import numpy as np
import information.information as inf
import _pickle
from plot.plot import plot
from threading import Lock


class InformationProcessor(object):
    def __init__(self, train, test, categories, filename=None, mi_estimator=None):
        """
        :param mi_estimator: mutual information estimator refer to parameters.py
        :param train: (x_train, y_train)
        :param test: (x_test, y_test)
        :param filename: what name to use to save images and data
        :param categories: how many distinct labels data has
        """

        self.mi_estimator = mi_estimator
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test
        self.x_full = np.concatenate((self.x_train, self.x_test))
        self.y_full = np.concatenate((self.y_train, self.y_test))
        self.categories = categories
        self.filename = filename
        self.mi = []
        self.__lock = Lock()
        self.__calculator = inf.calculate_information(self.x_full, self.y_full, self.mi_estimator)

    def information_calculator(self, activations):
        mutual_information = self.__calculator(activations)
        self.__lock.acquire()
        self.mi.append(mutual_information)
        self.__lock.release()

    def save(self, append=""):
        _pickle.dump(self.mi, open("output/data/" + self.filename + append, 'wb'))

    def plot(self, append="", show=False):
        i_x_t, i_y_t = zip(*self.mi)
        plot(i_x_t, i_y_t, "output/images/" + self.filename + append, show)




