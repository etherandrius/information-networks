import argparse
import _pickle
from plot.plot import plot_main


def output(args):
    if args.output is not None:
        return args.output
    out = args.input.split('/')[-1].split('.')[0]
    out = out[:-7] if out.endswith('_pickle') else out
    return "output/images/" + out + ".png"


def plot_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='input pickle data file')

    parser.add_argument('--output', dest='output', default=None,
                        help='output image file')

    args = parser.parse_args()
    args.output = output(args)
    return args


def main():
    args = plot_parameters()
    # data -> epoch * (i_x_t, i_y_t, i_t_t)
    data = _pickle.load(open(args.input, 'rb'))
    mi = list(zip(*map(lambda el: (el[0], *el[1]), data.items())))
    epochs, i_x_t, i_y_t, i_t_t = mi
    plot_main(i_x_t, i_y_t, epoch=max(epochs), filename=args.output, show=True)


if __name__ == '__main__':
    main()
