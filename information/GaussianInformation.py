import math


def local_likelihood():
    μ, Σ = 0, 0
    return μ, Σ


def entropy_of_data(u):
    h = 0
    for x in u:
        μ, Σ = local_likelihood()
        f = N(u; μ, Σ)
        h = h - (math.log2(f))/len(u)

    return h


def calculate_information_data(data_x, data_y):
    x = data_x
    y = data_y

    h_x = entropy_of_data(x)
    h_y = entropy_of_data(y)
    h_x_y = entropy_of_data(zip(x, y))

    mutual_information = h_x + h_y - h_x_y
    return mutual_information
