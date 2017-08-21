from __future__ import absolute_import

import numpy as np


def SGD(start, gradient, n, args=(), kwargs={}, learning_rate=0.1, momentum=0.5, maxepoch=250):
    """
    SGD to minimize sum(i=0 -> n) (1/n) * f(x)
    :param start: np.array(n)
    :param gradient:
    :param n:
    :param learning_rate:
    :param momentum:
    :param maxepoch:
    :param args: () arguments for the gradient
    :param kwargs:
    :return: np.array(n)
    """
    point = start
    v = np.zeros(len(start))
    for i in xrange(maxepoch):
        for j in xrange(n):
            v = momentum * v + learning_rate * gradient(point, *args, **kwargs)
            point -= v

    return point
