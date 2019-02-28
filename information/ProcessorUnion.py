from information.Processor import InformationProcessorDeltaExact


class InformationProcessorUnion(InformationProcessorDeltaExact):
    def __init__(self, ips):
        assert(len(ips) > 0)
        super().__init__(None)
        self.ips = ips

    def calculate_information(self, activation, epoch):
        for ip in self.ips:
            ip.calculate_information(activation, epoch)

    def finish_information_calculation(self):
        for ip in self.ips:
            ip.finish_information_calculation()

    def save(self, path):
        for (i, ip) in enumerate(self.ips):
            ip.save(path=path + "_{}".format(i))

    def plot(self, path, show=False):
        for (i, ip) in enumerate(self.ips):
            ip.plot(path=path + "_{}".format(i), show=show)


