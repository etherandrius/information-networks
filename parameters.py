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
        self.bins = args.bins
        self.epochs = args.epochs
        self.delta = args.delta
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

    parser.add_argument('--delta',
                        '-d', dest='delta', default=0.1,
                        type=float, help="Tolerance on how densely to calculate mutual information")

    parser.add_argument('--network_shape', '-ns', dest='shape', default="10,8,6,4",
                        help='Shape of the DNN, ex :'
                        '12,Dr,10-tanh,8-relu,6-sigmoid,BN,2 , would represent a DNN shape where 1st layer is Dense of size 12, 2nd layer is a Dropout layer, 3rd layer is Dense with size 10 and tanh activation function, 5th is Dense with relu activation function,..., 7th is BatchNormalization layer,..., note: 0th and last layers are automatically created to fit the dataset'
                        )

    parser.add_argument('--cores',
                        '-c', dest='cores', default=int(multiprocessing.cpu_count() / 2),
                        type=int,
                        help='How many cores to use for mutual information computation defaults to number of cores on the machine')

    parser.add_argument('--mi_estimator',
                        '-mie', dest='mi_estimator', default="Tishby",
                        help="Choose what mutual information estimator to use available: {}, ".format(estimators) +
                             "Tishby - method used by Tishby in his paper, "
                             "KDE - Kernel density estimator")

    parser.add_argument('--bins',
                        '-b', dest='bins', default=-1, type=int,
                        help="select number of bins to use for MIE's. -1 for no binning. Note: Tishby MIE requires binning and defaults to 30")

    parser.add_argument('--activation_function',
                        '-af', dest='activation', default="tanh",
                        help="Choose what neural network activation function to use available: {}".format(functions))

    parser.add_argument('--fabricated_dimmensions',
                        '-fd', dest="fab_dim", default=2,
                        type=int,
                        help="only relevant if data_set=Fabricated, how many irrelevant dimmensions to add to the input for exmample if input is dimension d and -fd=2 new input will have dimmesnion d+2 ")

    parser.add_argument('--fabricated_base',
                        '-fb', dest="fab_base", default="Tishby",
                        help="only relevant if data_set=Fabricated, what data set to use as a base for the fabricated data set default=Tishby, available: {}".format(
                            data_sets[:-1]))

    args = parser.parse_args()
    if args.fab_base == "Fabricated":
        raise ValueError("Fabricated cannot be a base for Fabricated")

    # args.shape = list(map(int, args.shape.split(',')))
    return Parameters(args)
