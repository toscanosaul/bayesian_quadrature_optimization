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
    EI_METHOD,
    SBO_METHOD,
)
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)

_aggregated_results = 'results_{problem_name}_{training_name}_{n_points}_{method}.json'.format
_aggregated_results_plot = 'plot_{problem_name}_{training_name}_{n_points}.pdf'.format


def plot_aggregate_results(multiple_spec, negative=True, square=True, title_plot=None,
                           y_label=None, n_iterations=None, repeat_ei=1):
    """

    :param multiple_spec: (multiple_spec entity) Name of the files with the aggregate results
    :return:
    """

    problem_names = list(set(multiple_spec.get('problem_names')))
    training_names = set(multiple_spec.get('training_names'))
    n_trainings = set(multiple_spec.get('n_trainings'))
    methods = set(multiple_spec.get('method_optimizations'))


    results = {}
    file_path_plot = None
    for problem in problem_names:
        if problem == 'aircraft_ei':
            continue
        dir = path.join(PROBLEM_DIR, problem, AGGREGATED_RESULTS)
        if not os.path.exists(dir):
            continue
        for training in training_names:
            for n_training in n_trainings:
                file_name = _aggregated_results_plot(
                    problem_name=problem,
                    training_name=training,
                    n_points=n_training,
                )

                if file_path_plot is None:
                    logger.info('problem is: %s' % dir)
                    file_path_plot = path.join(dir, file_name)
                for method in methods:
                    if method in results:
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

                    if repeat_ei > 1 and method == EI_METHOD:
                        new_x = []
                        for i in x_axis:
                            new_x += range(i * repeat_ei, (i + 1) * repeat_ei)
                        x_axis = new_x

                    if n_iterations is not None:
                        x_axis = x_axis[0:n_iterations]

                    y_values = []
                    ci_u = []
                    ci_l = []

                    for i in x_axis:
                        if repeat_ei > 1 and method == EI_METHOD:
                            j = i / repeat_ei
                        else:
                            j = i
                        y_values.append(data[str(j)]['mean'])
                        ci_u.append(data[str(j)]['ci_up'])
                        ci_l.append(data[str(j)]['ci_low'])

                    results[method] = [x_axis, y_values, ci_u, ci_l]

    colors = ['b', 'r', 'g', 'm']

    plt.figure()

    for id, method in enumerate(results):
        label = str(method)
        if label == SBO_METHOD:
            label = 'bqo'

        if label == 'ei':
            label = 'noisy_ei'
        x_axis = results[method][0]
        y_values = results[method][1]
        ci_u = results[method][2]
        ci_l = results[method][3]
        col = colors[id]
        plt.plot(x_axis, y_values, color=col, linewidth=2.0, label=label)
        plt.plot(x_axis, ci_u, '--', color=col, label="95% CI")
        plt.plot(x_axis, ci_l, '--', color=col)


    if title_plot is None:
        title_plot = problem_names[0]

    if y_label is None:
        y_label = 'Cross Validation Error'

    y_label = 'Fuel burn'

    plt.xlabel('Number of Samples', fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    plt.legend(loc=3, ncol=2, mode="expand", borderaxespad=0.)
#    plt.title(title_plot, fontsize=22)
    plt.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.12)
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(file_path_plot, bbox_inches = "tight")
