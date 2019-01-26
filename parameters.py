import argparse
import multiprocessing
from information.information import supported_estimators as estimators
from data.data import supported_data_sets as data_sets
from networks.networks import activation_functions as functions


class Fabricated(object):
    def __init__(self, args):
        self.dim = args.fab_dim
        self.base = args.fab_base


class Parameters(object):
    def __init__(self, args):
        self.train_size = args.train_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.skip = args.skip
        self.shape = args.shape
        self.cores = args.cores
        self.data_set = args.data_set
        self.mi_estimator = args.mi_estimator
        self.activation = args.activation
        self.fabricated = Fabricated(args)


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set',
                        '-ds', dest='data_set', default='Tishby',
                        help='choose a data set, available: {}'.format(data_sets) +
                             ', Tishby - data set used by Tishby in the original paper')

    parser.add_argument('--train_size',
                        '-ts', dest='train_size', default=0.8,
                        type=float, help='Training size')

    parser.add_argument('--batch_size',
                        '-bs', dest='batch_size', default=512,
                        type=int)

    parser.add_argument('--information_batch_number',
                        '-bn', dest='batch_number', default=30,
                        type=int, help='Number of batches to be used for information calculation')

    parser.add_argument('--num_of_epochs',
                        '-e', dest='epochs', default=1500,
                        type=int, help='Number of times to scan the dataset for NN training')

    parser.add_argument('--skip',
                        '-s', dest='skip', default=1,
                        type=int, help="Calculate information for every n'th mini-batch epoch")

    parser.add_argument('--network_shape', '-ns', dest='shape', default="12,10,8,6,4,2,1,2", help='Shape of the DNN')

    parser.add_argument('--cores',
                        '-c', dest='cores', default=multiprocessing.cpu_count(),
                        type=int,
                        help='How many cores to use for mutual information computation defaults to number of cores on the machine')

    parser.add_argument('--mi_estimator',
                        '-mie', dest='mi_estimator', default="bins",
                        help="Choose what mutual information estimator to use available: {}, ".format(estimators) +
                             "bins - method used by Tishby in his paper, (bins == bins-30)"
                             "bins-n - bins method but with n bins (ex. bins-10)"
                             "KDE - Kernel density estimator, "
                             "KSG - KSG mutual information estimator"
                             "KL - Kozachenko-Leonenko estimator, "
                             "LNN_1, LNN_2 - Local nearest neighbour with order 1 or 2")

    parser.add_argument('--activation_function',
                        '-af', dest='activation', default="tanh",
                        help="Choose what neural network activation function to use available: {}".format(functions))

    parser.add_argument('--fabricated_dimmensions',
                        '-fd', dest="fab_dim", default=2,
                        type=int, help="only relevant if data_set=Fabricated, how many irrelevant dimmensions to add to the input for exmample if input is dimension d and -fd=2 new input will have dimmesnion d+2 ")

    parser.add_argument('--fabricated_base',
                        '-fb', dest="fab_base", default="Tishby",
                        help="only relevant if data_set=Fabricated, what data set to use as a base for the fabricated data set default=Tishby, available: {}".format(data_sets[:-1]))

    args = parser.parse_args()
    if args.fab_base == "Fabricated":
        raise ValueError("Fabricated cannot be a base for Fabricated")

    #args.shape = list(map(int, args.shape.split(',')))
    return Parameters(args)
