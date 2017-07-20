from __future__ import absolute_import

from os import path
import numpy as np

from plots.plot_cv_validation_gp import (
    plot_diagnostic_plot,
    plot_histogram,
)
from stratified_bayesian_optimization.models.gp_fitting_gaussian import ValidationGPModel
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
    TASKS,
    DIAGNOSTIC_KERNEL_DIR,
)
from stratified_bayesian_optimization.util.json_file import JSONFile


if __name__ == '__main__':
    # usage: python -m scripts.validation_kernel_plots

    # script used to generate plots of the histogram and diagnostic plot for the validation of the
    # kernel


    random_seed = 5
    n_training = 3
    problem_name = "movies_collaborative"
    type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
    same_correlation = False

    kernel_name = ''
    for kernel in type_kernel:
        kernel_name += kernel + '_'
    kernel_name = kernel_name[0: -1]
#    kernel_name += '_same_correlation_' + str(same_correlation)


    diag_dir = path.join(DIAGNOSTIC_KERNEL_DIR, problem_name)

    filename = path.join(diag_dir, ValidationGPModel._validation_filename(
        problem=problem_name,
        type_kernel=kernel_name,
        n_training=n_training,
        random_seed=random_seed,
    ))

    data = JSONFile.read(filename)

    plot_histogram(np.array(data['y_eval']), np.array(data['means']), np.array(data['std_vec']),
                   data['filename_histogram'])
    plot_diagnostic_plot(np.array(data['y_eval']), np.array(data['means']),
                         np.array(data['std_vec']), data['n_data'],
                         data['filename_plot'])
