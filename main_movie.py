from plot.plot import plot_movie
import sys
import argparse
import _pickle
import numpy as np

def main():
    args = params()
    # data -> epoch * (i_x_t, i_y_t, i_t_t)
    data = _pickle.load(open(args.input, 'rb'))
    data = list(zip(*data))
    # i_t_t might not exist
    i_x_t, i_y_t = data[0], data[1]
    plot_movie(i_x_t, i_y_t, args, filename=args.output)


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        '-i', dest='input', required=True,
                        help='input pickle data file')

    parser.add_argument('--output',
                        '-o', dest='output', default=None,
                        help='output movie file')

    parser.add_argument('--delta',
                        '-d', dest='delta', default=0.05, type=float,
                        help='delta tuning skipping of epochs')

    parser.add_argument('--epoch_multiplier',
                        '-em', dest='em', default=1, type=int,
                        help='if em canot be extracted from the filepath this em will be used')

    parser.add_argument('--movie_length',
                        '-ml', dest='movie_length', default=40, type=int,
                        help='approx movie length in seconds')

    args = parser.parse_args()
    args.output = output(args)
    args.em = extract_em(args)

    return args


def extract_em(args):
    em = args.input
    em = em.split(',s-')
    if len(em) >= 2:
        em = int(em[1].split(',')[0])
    else:
        em = args.em
    return em


def output(args):
    if args.output is not None:
        return args.output
    out = args.input.split('/')[-1].split('.')[0]
    out = out[:-7] if out.endswith('_pickle') else out
    return "output/movies/" + out



if __name__ == '__main__':
    main()

