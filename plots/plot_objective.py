from __future__ import absolute_import

import matplotlib.pyplot as plt

import numpy as np
from os import path

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.entities.objective import Objective
from stratified_bayesian_optimization.lib.constant import (
    PARTIAL_RESULTS,
    PROBLEM_DIR,
)

logger = SBOLog(__name__)

_filename_obj_af_plots =  'plot_objective_af_{problem_name}.png'.format
_filename_obj_plots =  'plot_objective_{problem_name}.png'.format


def plot_objective_function_af(problem_name, filename_plot, bounds, n_points_by_dimension=None,
                               n_samples=0, n_tasks=0):
    """
    Plot the objective function used for the acquisition function.

    :param problem_name: (str)
    :param filename_plot: (str)
    :param bounds: [[float, float]] (only for x domain)
    :param n_points_by_dimension: (int)
    :param n_samples: (int)
    :param n_tasks: (int)
    """


    n_points = n_points_by_dimension
    if n_points is None:
        n_points = (bounds[0][1] - bounds[0][0]) * 10


    points = np.linspace(bounds[0][0], bounds[0][1], n_points)

    name_module = TrainingDataService.get_name_module(problem_name)
    module = __import__(name_module, globals(), locals(), -1)

    values = {}
    filename_plot = filename_plot[0 : -4]

    if n_tasks > 0:
        for i in xrange(n_tasks):
            vals = []
            for point in points:
                point_ = np.concatenate((np.array([point]), np.array([i])))
                evaluation = TrainingDataService.evaluate_function(module, point_, n_samples)
                vals.append(evaluation)
            values[i] = vals
            plt.figure()
            plt.plot(points, values[i], label='task_'+str(i))
            plt.legend()

            plt.savefig(filename_plot + '_task_'+str(i) + '.png')

def plot_objective_function(problem_name, filename_plot, bounds, n_points_by_dimension=None,
                               n_samples=0):
    """
    Plot the objective function used for the acquisition function.

    :param problem_name: (str)
    :param filename_plot: (str)
    :param bounds: [[float, float]] (only for x domain)
    :param n_points_by_dimension: (int)
    :param n_samples: (int)
    """

    n_points = n_points_by_dimension
    if n_points is None:
        n_points = (bounds[0][1] - bounds[0][0]) * 10


    points = np.linspace(bounds[0][0], bounds[0][1], n_points)

    name_module = TrainingDataService.get_name_module(problem_name)
    module = __import__(name_module, globals(), locals(), -1)

    values = []
    for point in points:
        evaluation = Objective.evaluate_objective(module, [point], n_samples)
        values.append(evaluation)
    plt.figure()
    plt.plot(points, values, label='objective')
    plt.legend()
    plt.savefig(filename_plot)


def make_plots(problem_name, n_tasks, bounds, n_samples=0):
    """
    Generates evaluations of the posterior mean, and write them in the debug directory.

    :param problem_name: (str)
    :param n_tasks: (int)
    :param bounds: [[float, float]] (only for x domain)
    :param n_samples: (int)
    """

    obj_dir = path.join(PROBLEM_DIR, problem_name)

    if n_tasks > 0 :
        f_name = _filename_obj_af_plots(problem_name=problem_name)
        plot_file =  path.join(obj_dir, f_name)

        plot_objective_function_af(problem_name, plot_file, bounds,
                                   n_samples=n_samples, n_tasks=n_tasks)

    f_name = _filename_obj_plots(problem_name=problem_name)
    plot_file =  path.join(obj_dir, f_name)

    plot_objective_function(problem_name, plot_file, bounds, n_samples=n_samples)

