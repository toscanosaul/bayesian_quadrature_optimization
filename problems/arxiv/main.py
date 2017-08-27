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


# Training data has 246650 observations
year = '2016'
month = '1'
n_folds = 5
# num_item = 5000
# num_user = 4815
num_item = 2018
num_user = 2752
# there are 40306 observations

# num_item = 326
# num_user = 507
# there are 90271 observations

train=[]
validate=[]

file_name = TrainingData._name_fold_indexes(year=year, month=month)
random_indexes = JSONFile.read(file_name)

# file_name = TrainingData._name_training_data(year=year, month=month)
# training_data = JSONFile.read(file_name)

for i in range(n_folds):
    file_name = TrainingData._name_fold_data_training(year=year, month=month, fold=i)
    training = JSONFile.read(file_name)
    train.append(np.array(training))

    file_name = TrainingData._name_fold_data_validation(year=year, month=month, fold=i)
    validation = JSONFile.read(file_name)
    validate.append(np.array(validation))

def toy_example(x):
    """

    :param x: [float, float, int, int, int]
    :return: [float]
    """
    epsilon = x[0]
    lamb = x[1]
    maxepoch = max(int(x[3]), 1)
    num_feat = max(int(x[2]), 1)
    task = int(x[4])

    # validation = [training_data[index] for index in random_indexes[task]]
    # validation = matlab.double(validation)
    #
    # training_indexes = []
    # for j in xrange(n_folds):
    #     if j != task:
    #         training_indexes += random_indexes[j]
    #
    # training = [training_data[index] for index in training_indexes]
    # training = matlab.double(training)

    training = train[task]
    validation = validate[task]

    val = PMF(num_user, num_item, training, validation, epsilon, lamb, maxepoch, num_feat,
              l_rating=1, u_rating=2)
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
        # val = toy_example(point)
        # values.append(val[0])

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
