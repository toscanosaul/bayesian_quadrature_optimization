from __future__ import absolute_import

import numpy as np

from copy import deepcopy

from problems.pmf.pmf import PMF


num_user = 943
num_item = 1682

train=[]
validate=[]

for i in range(1,6):
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
    values = []
    for task in xrange(5):
        point = deepcopy(x)
        point.append(task)
        val = toy_example(point)
        values.append(val[0])

    return [np.mean(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)

