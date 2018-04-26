from __future__ import absolute_import

import numpy as np
import argparse

from multi_start.parametric_model import ParametricModel
from multi_start.stat_model import StatModel

from stratified_bayesian_optimization.util.json_file import JSONFile


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('starting_point', help='e.g. 0')

    args = parser.parse_args()

    starting_point_index = int(args.starting_point)

    dir_data = 'data/multi_start/neural_networks/training_results/'

    n_epochs = 20
    n_batches = 60
    total_iterations = n_epochs * n_batches

    cnn_data = {}
    cnn_data[starting_point_index] = JSONFile.read(dir_data + str(starting_point_index))

    for j in cnn_data[starting_point_index]:
        cnn_data[starting_point_index][j] = [t / 100.0 for t in cnn_data[starting_point_index][j]]

    def get_values(i, index):
        data = cnn_data[index]
        return data[str(i / (n_batches + 1) + 1)][(i - 1) % n_batches]

    training_data = {}
    best_results = {}
    functions_get_value = {}
    arguments = {}
    n_training = 3

    training_data[starting_point_index] = cnn_data[starting_point_index][str(1)][0: n_training]
    best_results[starting_point_index] = np.max(training_data[starting_point_index])
    functions_get_value[starting_point_index] = get_values
    arguments[starting_point_index] = {'index': starting_point_index}

    stat_models = {}
    i = starting_point_index
    model = StatModel(
        training_data[i], best_results[i], n_training, functions_get_value[i], None, n_training,
        1, kwargs_get_value_next_iteration=arguments[i], problem_name='cnn',
        max_iterations=total_iterations, parametric_mean=False, lower=0.0, upper=1.0,
        total_batches=n_batches)
    stat_models[i] = model

    i = starting_point_index
    accuracy_results = {}
    model = stat_models[i]
    accuracy_results[i] = model.accuracy(
        model.gp_model, start=n_training, iterations=100, sufix=str(i))

    model.plot_accuracy_results(
        accuracy_results[i][0], accuracy_results[i][1], cnn_data[i][str(n_epochs)][-1],
        start=n_training, sufix=str(i))
