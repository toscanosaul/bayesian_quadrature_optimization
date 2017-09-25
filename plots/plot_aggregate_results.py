from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

import numpy as np
from os import path
import os

from stratified_bayesian_optimization.lib.constant import (
    DEFAULT_RANDOM_SEED,
    UNIFORM_FINITE,
    DOGLEG,
    PROBLEM_DIR,
    PARTIAL_RESULTS,
    AGGREGATED_RESULTS,
)
from stratified_bayesian_optimization.util.json_file import JSONFile

_aggregated_results = 'results_{problem_name}_{training_name}_{n_points}_{method}.json'.format
_aggregated_results_plot = 'plot_{problem_name}_{training_name}_{n_points}.pdf'.format


def plot_aggregate_results(multiple_spec, negative=True, square=True):
    """

    :param multiple_spec: (str) Name of the file with the aggregate results
    :return:
    """

    problem_names = set(multiple_spec.get('problem_names'))
    training_names = set(multiple_spec.get('training_names'))
    n_trainings = set(multiple_spec.get('n_trainings'))
    methods = set(multiple_spec.get('method_optimizations'))

    results = {}
    for problem, training, n_training, method in zip(problem_names, training_names,
                                                     n_trainings, methods):

        dir = path.join(PROBLEM_DIR, problem, AGGREGATED_RESULTS)

        if not os.path.exists(dir):
            continue

        file_name = _aggregated_results(
            problem_name=problem,
            training_name=training,
            n_points=n_training,
            method=method,
        )

        file_path = path.join(dir, file_name)

        if not os.path.exists(file_path):
            continue

        data = JSONFile.read(file_path)

        x_axis = list(data.keys())
        x_axis = [int(i) for i in x_axis]
        x_axis.sort()

        y_values = []
        ci_u = []
        ci_l = []

        for i in x_axis:
            y_values.append(data[str(i)]['mean'])
            ci_u.append(data[str(i)]['ci_up'])
            ci_l.append(data[str(i)]['ci_low'])

        results[method] = [x_axis, y_values, ci_u, ci_l]

        file_name = _aggregated_results_plot(
            problem_name=problem,
            training_name=training,
            n_points=n_training,
        )

        file_path = path.join(dir, file_name)

    colors = ['b', 'r']

    for id, method in enumerate(results):
        x_axis = results[method][0]
        y_values = results[method][1]
        ci_u = results[method][2]
        ci_l = results[method][3]
        plt.figure()
        col = colors[id]
        plt.plot(x_axis, y_values, color=col, linewidth=2.0, label=str(method))
        plt.plot(x_axis, ci_u, '--', color=col, label="95% CI")
        plt.plot(x_axis, ci_l, '--', color=col)

    plt.xlabel('Number of Samples', fontsize=24)
    plt.ylabel('Cross Validation Error', fontsize=24)
    plt.legend(loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(file_path)
