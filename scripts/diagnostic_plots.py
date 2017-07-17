from __future__ import absolute_import

from plots.plot_acquisition_function_posterior_mean import make_plots
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
    TASKS,
)


if __name__ == '__main__':
    # usage: python -m scripts.diagnostic_plots

    # script used to generate plots of the acquisition function and posterior mean

    problem_name = 'test_simulated_gp'
    model_type = 'gp_fitting_gaussian'
    training_name = 'test_sbo'
    n_training = 5
    random_seed = 5
    kernel_type = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
    n_tasks = 2
    n_iterations = 5
    make_plots(problem_name, model_type, training_name, n_training, random_seed,
               kernel_type, n_tasks, n_iterations)


