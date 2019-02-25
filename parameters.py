import argparse
from information.information import supported_estimators as estimators
from data.data import supported_data_sets as data_sets
from networks.networks import activation_functions as functions


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


def general_parameters(parser):
    parameters = parser.add_argument_group('General parameters')
    pa.add_argument('--delta',
                        '-d', dest='delta', default=0.1,
                        type=float, help="Tolerance on how densely to calculate mutual information")

    parameters.add_argument('--cores',
                        '-c', dest='cores', default=1,
                        type=int,
                        help='number of information instances to compute at a time')

    pa.add_argument('--mi_estimator',
                        '-mie', dest='mi_estimator', default="Tishby",
                        help="Choose what mutual information estimator to use available: {}, ".format(estimators) +
                             "Tishby - method used by Tishby in his paper, "
                             "KDE - Kernel density estimator")

    parser.add_argument('--bins',
                        '-b', dest='bins', default=-1, type=int,
                        help="select number of bins to use for MIE's. -1 for no binning. Note: Tishby MIE requires binning and defaults to 30")

    args = parser.parse_args()
    # args.shape = list(map(int, args.shape.split(',')))
    return Parameters(args)
