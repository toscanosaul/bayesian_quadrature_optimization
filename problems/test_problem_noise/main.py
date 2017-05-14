from __future__ import absolute_import

import numpy as np


def toy_example(n_samples, x):
    """
    :param n_samples: int
    :param x: [[float]], The first part of the vector represents always x, and the second part is w
        (if w is considered in the problem).

    :return: [float, float]
    """
    noise = np.random.normal(0, 1, n_samples)

    evaluations = x + noise


    return [np.mean(evaluations), np.var(evaluations) / (n_samples ** 2)]

def main(n_samples, *params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)

    return toy_example(n_samples, *params)