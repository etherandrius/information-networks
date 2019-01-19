from data.InformationProcessor import InformationProcessor


class InformationProcessorFabricatedData(InformationProcessor):
    def __init__(self, train, test, categories,  num_relevant_dim, filename=None, mi_estimator=None):
        """
        :param num_relevant_dim: how many dimensions are relevant input, if our x has 10 dimensions and
                                 num_relevant_dim = 2 then only first two of them are relevant others are irrelevant
        """
        super().__init__(train, test, categories)
        x_train, y_train = train
        x_test, y_test = test

        x_train_rel = x_train[:, :num_relevant_dim]
        x_train_irr = x_train[:, num_relevant_dim:]
        x_test_rel = x_test[:, :num_relevant_dim]
        x_test_irr = x_test[:, num_relevant_dim:]

        self._relIP = InformationProcessor(
            (x_train_rel, y_train), (x_test_rel, y_test), categories, filename + "_rel", mi_estimator)
        self._irrIP = InformationProcessor(
            (x_train_irr, y_train), (x_test_irr, y_test), categories, filename + "_irr", mi_estimator)

    def information_calculator(self, activations):
        self._relIP.information_calculator(activations)
        self._irrIP.information_calculator(activations)

    def save(self):
        self._relIP.save()
        self._irrIP.save()

    def plot(self, show=False):
        self._relIP.plot(show)
        self._irrIP.plot(show)



