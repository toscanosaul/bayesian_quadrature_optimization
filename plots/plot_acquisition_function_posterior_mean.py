from __future__ import absolute_import

import matplotlib.pyplot as plt

import numpy as np
from os import path
import os

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    DEBUGGING_DIR,
)
from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature

logger = SBOLog(__name__)

colors = ['b','g','r','c','m','y','k']

_filename_voi_plots =  '{iteration}_sbo_plot_{model_type}_{problem_name}_' \
                            '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                            'png'.format

_filename_mu_plots = '{iteration}_post_mean_gp_plot_{model_type}_{problem_name}_' \
                           '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                           'png'.format

def plot_acquisition_function(filename_af, filename_plot, n_tasks=0):
    """

    :param filename_af: (str) Filename of the plot with the acquisition function evaluations
    :param filename_plot: (str) Filename used to save the plots
    :param n_tasks: (int)
    """

    data = JSONFile.read(filename_af)
    evaluations = data['evaluations']
    points = data['points']
    if n_tasks > 0:
        filename_plot = filename_plot[0: -4]
        for i in xrange(n_tasks):
            plt.figure()
            evals = evaluations[str(i)]
            plt.plot(points, evals, label='task_'+str(i))
            plt.legend()
            plt.savefig(filename_plot + 'task_'+str(i) + '.png')

def plot_posterior_mean(filename_af, filename_plot):
    """

    :param filename_af: (str) Filename of the plot with the acquisition function evaluations
    :param filename_plot: (str) Filename used to save the plots
    """

    data = JSONFile.read(filename_af)
    evaluations = data['evaluations']
    points = data['points']

    plt.figure()
    evals = np.array([eval[0] for eval in evaluations])
    plt.plot(points, evals)
    plt.savefig(filename_plot)

def make_plots(problem_name, model_type, training_name, n_training, random_seed,
               kernel_type, n_tasks, n_iterations):
    """
    Generates evaluations of the posterior mean, and write them in the debug directory.

    :param problem_name: (str)
    :param model_type: (str)
    :param training_name: (str)
    :param n_training: (int)
    :param random_seed: (int)
    :param kernel_type: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL].
            If we want to use a scaled NAME_1_KERNEL, the parameter must be
            [SCALED_KERNEL, NAME_1_KERNEL].
    :param n_tasks: (int)
    :param n_iterations: (int)

    """


    debug_dir = path.join(DEBUGGING_DIR, problem_name)

    kernel_name = ''
    for kernel in kernel_type:
        kernel_name += kernel + '_'
    kernel_name = kernel_name[0: -1]

    for iteration in xrange(n_iterations):
        f_name = SBO._filename_voi_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path_af = path.join(debug_dir, f_name)

        f_name_plot = _filename_voi_plots(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        file_plot = path.join(debug_dir, f_name_plot)

        plot_acquisition_function(debug_path_af, file_plot, n_tasks=n_tasks)

        f_name = BayesianQuadrature._filename_mu_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path_mu = path.join(debug_dir, f_name)

        f_name_plot = _filename_mu_plots(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        file_plot = path.join(debug_dir, f_name_plot)

        plot_posterior_mean(debug_path_mu, file_plot)
