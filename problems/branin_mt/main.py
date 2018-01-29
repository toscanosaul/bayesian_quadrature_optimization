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

# x_2 = {0.25: 0, 0.5: 1, 0.75: 2}
# x_3 = {0.2: 0, 0.4: 1, 0.6: 2, 0.8: 3}

x_2 = [0.25, 0.5, 0.75]
x_3 = [0.2, 0.4, 0.6, 0.8]

probabilities = np.array([[0.0375, 0.0875, 0.0875, 0.0375], [0.0750, 0.1750, 0.1750, 0.0750],
                          [0.0375, 0.0875, 0.0875, 0.0375]])


def branin(u, v):
    result = 10 + 10.0 * np.cos(u) * (1.0 - (1.0 / (8.0 * np.pi)))
    result += (v - (5.1) * (u**2) * (4.0 * np.pi * np.pi)**(-1) + 5.0 * u * (np.pi)**-1 - 6)**2
    return result

def toy_example(x):
    """

    :param x: [float, float, int]
    :return: [float]
    """
    x1 = x[0]
    x4 = x[1]
    index = int(x[2])

    ind_x2 = int(index / 4.0)
    ind_x3 = index % 4

    x2 = x_2[ind_x2]
    x3 = x_3[ind_x3]


    result = branin(15*x1-5,15*x2) * branin(15*x3-5,15*x4)

    return [result]

def integrate_toy_example(x):
    """

    :param x: [float, float]
    :return: [float]
    """

    values = []
    for i in range(12):
        ind_x1 = int(i / 4.0)
        ind_x2 = i % 4
        weight = probabilities[ind_x1, ind_x2]
        point = x + [i]
        values.append(weight * toy_example(point)[0])

    return [np.sum(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)