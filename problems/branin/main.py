from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.pmf.pmf import PMF
from problems.arxiv.generate_training_data import TrainingData
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_to_list,
)

x_2 = {0.25: 0, 0.5: 1, 0.75: 2}
x_3 = {0.2: 0, 0.4: 1, 0.6: 2, 0.8: 3}

probabilities = np.array([[0.0375, 0.0875, 0.0875, 0.0375], [0.0750, 0.1750, 0.1750, 0.0750],
                          [0.0375, 0.0875, 0.0875, 0.0375]])


def branin(u, v):
    result = 10 + 10.0 * np.cos(u) * (1.0 - (1.0 / (8.0 * np.pi)))
    result += (v - (5.1) * (u**2) * (4.0 * np.pi * np.pi)**(-1) + 5.0 * u * (np.pi)**-1 - 6)**2
    return result

def toy_example(x):
    """

    :param x: [float, float, int, int, int]
    :return: [float]
    """
    x1 = x[0]
    x4 = x[1]
    x2 = x[2]
    x3 = x[3]

    result = branin(15*x1-5, 15*x2) * branin(15*x3-5,15*x4)

    return [result]

def integrate_toy_example(x):
    """

    :param x: [float, float]
    :return: [float]
    """

    values = []
    for x2 in x_2.keys():
        for x3 in x_3.keys():
            ind1 = x_2[x2]
            ind2 = x_3[x3]
            weight = probabilities[ind1, ind2]
            point = x + [x2, x3]
            values.append(weight * toy_example(point)[0])

    return [np.sum(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)