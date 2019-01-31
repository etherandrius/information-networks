from information.Processor import InformationProcessor


class InformationProcessorFabricatedData(InformationProcessor):
    def __init__(self, train, test, categories, num_relevant_dim, filename=None, mi_estimator=None, delta=0.2,
                 max_workers=2, bins=30):
        """
        :param num_relevant_dim: how many dimensions are relevant input, if our x has 10 dimensions and
                                 num_relevant_dim = 2 then only first two of them are relevant others are irrelevant
        """
        super().__init__(train, test, categories, mi_estimator=mi_estimator)
        x_train, y_train = train
        x_test, y_test = test

        x_train_rel = x_train[:, :num_relevant_dim]
        x_train_irr = x_train[:, num_relevant_dim:]
        x_test_rel = x_test[:, :num_relevant_dim]
        x_test_irr = x_test[:, num_relevant_dim:]

        self._relIP = InformationProcessor((x_train_rel, y_train), (x_test_rel, y_test),
                                           categories, filename + "_rel", mi_estimator, delta, max_workers, bins)
        self._irrIP = InformationProcessor((x_train_irr, y_train), (x_test_irr, y_test),
                                           categories, filename + "_irr", mi_estimator, delta, max_workers, bins)

    def calculate_information(self, activation, epoch):
        self._relIP.calculate_information(activation, epoch)
        self._irrIP.calculate_information(activation, epoch)

    def finish_information_calculation(self):
        self._relIP.finish_information_calculation()
        self._irrIP.finish_information_calculation()

    def save(self, append=""):
        self._relIP.save(append=append)
        self._irrIP.save(append=append)

    def plot(self, append="", show=False):
        self._relIP.plot(show=show, append=append)
        self._irrIP.plot(show=show, append=append)



