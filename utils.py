import itertools


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)
