from __future__ import absolute_import

import numpy as np


def toy_example(x):
    """
    :param x: [[float]]. The first part of the vector represents always x, and the second part is w
        (if w is considered in the problem).
    :return: [float]
    """
    return [np.sum(x)]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)
