from plot.plot import plot_epoch
import sys
import _pickle
import numpy as np

path = sys.argv[1]
# data -> epoch * (i_x_t, i_y_t, i_t_t)
data = _pickle.load(open(path, 'rb'))
i_x_t, i_y_t, i_t_t = zip(*data)
plot_epoch(i_x_t, i_y_t)
print("A")



