from __future__ import absolute_import

import numpy as np

from copy import deepcopy

from problems.pmf.pmf import PMF
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_to_list,
)


num_user = 943
num_item = 1682

n_folds = 5

train=[]
validate=[]

for i in range(1, 6):
    data = np.loadtxt("problems/movies_collaborative/ml-100k/u%d.base"%i)
    test = np.loadtxt("problems/movies_collaborative/ml-100k/u%d.test"%i)
    train.append(data)
    validate.append(test)

def toy_example(x):
    """

    :param x: [float, float, int, int, int]
    :return: [float]
    """
    epsilon = x[0]
    lamb = x[1]
    maxepoch = int(x[3])
    num_feat = int(x[2])
    task = int(x[4])

    val = PMF(num_user, num_item, train[task], validate[task], epsilon, lamb, maxepoch, num_feat)
    return [val]

def integrate_toy_example(x):
    """

    :param x: [float, float, int, int]
    :return: [float]
    """

    points = {}
    for task in xrange(n_folds):
        point = deepcopy(x)
        point.append(task)
        points[task] = point

    errors = Parallel.run_function_different_arguments_parallel(
        toy_example, points)

    values = convert_dictionary_to_list(errors)

    return [np.mean(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)

