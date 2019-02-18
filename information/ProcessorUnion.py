from information.Processor import InformationProcessor


class InformationProcessorUnion(InformationProcessor):
    def __init__(self, ips):
        assert(len(ips) > 0)
        ip = ips[0]
        train = ip.x_train, ip.y_train
        test = ip.x_test, ip.y_test
        super().__init__(train, test, ip.categories)
        self.ips = ips
        for ip in ips:
            ip.x_train = None
            ip.y_train = None
            ip.x_test = None
            ip.y_test = None

    def calculate_information(self, activation, epoch):
        for ip in self.ips:
            ip.calculate_information(activation, epoch)

    def finish_information_calculation(self):
        for ip in self.ips:
            ip.finish_information_calculation()

    def save(self, append=""):
        for (i, ip) in enumerate(self.ips):
            ip.save(append=append + "_{}".format(i))

    def plot(self, append="", show=False):
        for (i, ip) in enumerate(self.ips):
            ip.plot(append=append + "_{}".format(i), show=show)


