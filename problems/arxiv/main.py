from __future__ import absolute_import

import numpy as np
import matlab.engine
from copy import deepcopy

from problems.pmf.pmf_matlab import PMF
from problems.arxiv.generate_training_data import TrainingData
from stratified_bayesian_optimization.util.json_file import JSONFile

year = '2016'
month = '1'
n_folds = 5
num_item = 5000
num_user = 4815

train=[]
validate=[]

file_name = TrainingData._name_fold_indexes(year=year, month=month)
random_indexes = JSONFile.read(file_name)

file_name = TrainingData._name_training_data(year=year, month=month)
training_data = JSONFile.read(file_name)

# for i in range(n_folds):
#     file_name = TrainingData._name_fold_data_training(year=year, month=month, fold=i)
#     training = JSONFile.read(file_name)
#     train.append(matlab.double(training))
#
#     file_name = TrainingData._name_fold_data_validation(year=year, month=month, fold=i)
#     validation = JSONFile.read(file_name)
#     validate.append(matlab.double(validation))

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

    validation = [training_data[index] for index in random_indexes[task]]
    validation = matlab.double(validation)

    training_indexes = []
    for j in xrange(n_folds):
        if j != task:
            training_indexes += random_indexes[j]

    training = [training_data[index] for index in training_indexes]
    training = matlab.double(training)

    val = PMF(num_user, num_item, training, validation, epsilon, lamb, maxepoch, num_feat,
              l_rating=1, u_rating=2)
    return [-1.0 * val ** 2]

def integrate_toy_example(x):
    """

    :param x: [float, float, int, int]
    :return: [float]
    """
    values = []
    for task in xrange(n_folds):
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
