from information.Processor import InformationProcessor


class InformationProcessorUnion(InformationProcessor):
    def __init__(self, ips):
        assert(len(ips) > 0)
        ip = ips[0]
        train = ip.x_train, ip.y_train
        test = ip.x_test, ip.y_test
        super().__init__(train, test, ip.categories)
        self.ips = ips
        # clearing up some memory
        for ip in ips:
            ip.x_train = None
            ip.y_train = None
            ip.x_test = None
            ip.y_test = None

    def information_calculator(self, activations):
        for ip in self.ips:
            ip.information_calculator(activations)

    def save(self, append=""):
        for ip in self.ips:
            ip.save(append=append)

    def plot(self, append="", show=False):
        for ip in self.ips:
            ip.plot(append=append, show=show)


