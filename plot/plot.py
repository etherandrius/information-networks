import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from tqdm import tqdm
from utils import pairwise

def plot_main(data_x, data_y, filename=None, show=False):
    print("Producing information plane image")
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    for ix in tqdm(range(len(data_x))):
        for e in pairwise(zip(data_x[ix], data_y[ix])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)

    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()
    plt.cla()


def plot_bilayer(series, filename=None, show=False):
    print("Producing bilayer information image")
    series = np.array(series).swapaxes(0, 1)
    for ix, layer in enumerate(series):
        plt.plot(layer, label="{} -> {}".format(ix, ix+1))

    plt.xlabel("Time")
    plt.ylabel("Mutual Information")
    plt.title("Mutual Information between layer i and i+1")
    plt.legend(loc="upper left")
    if filename is not None:
        print("Saving image to file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    if show:
        plt.show()
    plt.cla()


def plot_movie(data_x, data_y, filename=None, show=False):
    print("Producing movie of the information plane")
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    figure = plt.figure()
    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    for ix in range(len(data_x)):
        for e in pairwise(zip(data_x[ix], data_y[ix])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.savefig("output/{}".format(ix), dpi=100)


    if filename is not None:
        print("Saving movie to a file : ", filename)
        start = time.time()
        plt.savefig(filename, dpi=1000)
        end = time.time()
        print("Time taken to save to file {:.3f}s".format((end-start)))
    plt.cla()


def plot_epoch(data_x, data_y, filename=None, show=False):
    print("Producing information plane image")
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_x))]
    figure = plt.figure()
    plt.xlabel('I(X,T)')
    plt.ylabel('I(Y,T)')

    def single_epoch(ix):
        for e in pairwise(zip(data_x[ix], data_y[ix])):
            (x1, y1), (x2, y2) = e
            plt.plot((x1, x2), (y1, y2), color=colors[ix], alpha=0.9, linewidth=0.2, zorder=ix)
            point_size = 300
            plt.scatter(x1, y1, s=point_size, color=colors[ix], zorder=ix)
            plt.scatter(x2, y2, s=point_size, color=colors[ix], zorder=ix)
            figure.canvas.draw()
            figure.canvas.flush_events()
        return figure

    movie = anim.FuncAnimation(figure, single_epoch, frames=20)
    plt.show()



    #if filename is not None:
    #    print("Saving image to file : ", filename)
    #    start = time.time()
    #    plt.savefig(filename, dpi=1000)
    #    end = time.time()
    #    print("Time taken to save to file {:.3f}s".format((end-start)))
    #if show:
    #    plt.show()
    #plt.cla()




