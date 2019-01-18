from data.InformationProcessor import InformationProcessor
import information.information as inf


class InformationProcessorFabricatedData(InformationProcessor):
    def __init__(self, mi_estimator, train, test, categories, num_relevant_dim):
        """
        :param num_relevant_dim: how many dimensions are relevant input, if our x has 10 dimensions and num_relevant_dim = 2 then only first two of them are relevant others are irrelevent
        """
        super().__init__(mi_estimator, train, test, categories)
        x_train, y_train = train
        x_test, y_test = test

        x_train_rel = x_train[:, :num_relevant_dim]
        x_train_irr = x_train[:, num_relevant_dim:]
        x_test_rel = x_test[:, :num_relevant_dim]
        x_test_irr = x_test[:, num_relevant_dim:]

        self._relIP = InformationProcessor(mi_estimator, (x_train_rel, y_train), (x_test_rel, y_test), categories)
        self._irrIP = InformationProcessor(mi_estimator, (x_train_irr, y_train), (x_test_irr, y_test), categories)

    def information_calculator(self, activations):
        self._relIP.information_calculator(activations)
        self._irrIP.information_calculator(activations)

    def save(self, path):
        self._relIP.save(path + "_rel")
        self._irrIP.save(path + "_irr")

    def plot(self, path=None, show=False):
        self._relIP.plot(path + "_rel", show)
        self._irrIP.plot(path + "_irr", show)



