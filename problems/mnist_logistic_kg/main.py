from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.mnist_logistic_kg.logistic import train_logistic

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_to_list,
)


def toy_example(x):
    """

    :param x: [float, float, int, int, int, task]
    :return: [float]
    """
    momentum = x[0]
    lr = x[1]
    batch_size = int(x[2])
    alpha = x[3]
    maxepoch = int(x[4])

    val = train_logistic(momentum=momentum, lr=lr, batch_size=batch_size, alpha=alpha,
                         maxepoch=maxepoch)

    return [-1.0 * val[2]]

def integrate_toy_example(x):
    """

    :param x: [float, float, int, int]
    :return: [float]
    """

    return toy_example(x)


def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)