import ujson

import numpy as np

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    SBO_METHOD,
    MULTI_TASK_METHOD,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions


if __name__ == '__main__':
    # usage: python -m scripts.generate_spec > data/specs/test_spec.json

    # script used to generate spec file to run BGO

    dim_x = 4
    bounds_domain_x = [(0.01, 1.01), (0.1, 2.1), (1, 21), (1, 201)]
    problem_name = 'movies_collaborative'
    training_name = None
    type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
    dimensions = [5, 4, 5]
    bounds_domain = [[0.01, 1.01], [0.1, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    n_training = 50
    random_seed = 7
    type_bounds = [0, 0, 0, 0, 1]
    x_domain = [0, 1, 2, 3]
    number_points_each_dimension = [6, 6, 11, 6]
    mle = True
    distribution = UNIFORM_FINITE
    parallel = True
    thinning = 5
    n_burning = 100
    max_steps_out = 1000
    n_iterations = 100
    same_correlation = True
    debug = False
    number_points_each_dimension_debug = [10, 10, 10, 10]
    monte_carlo_sbo = True
    n_samples_mc = 20
    n_restarts_mc = 500
    n_best_restarts_mc = 10
    factr_mc = 1e12
    maxiter_mc = 10
    n_restarts = 50
    n_best_restarts = 5
    use_only_training_points = True
    method_optimization = SBO_METHOD
    n_samples_parameters = 15
    n_restarts_mean = 1000
    n_best_restarts_mean = 100

    spec = SpecService.generate_dict_spec(
        problem_name, dim_x, bounds_domain_x, training_name, type_kernel, dimensions,
        bounds_domain=bounds_domain, n_training=n_training, random_seed=random_seed,
        type_bounds=type_bounds, x_domain=x_domain, mle=mle,
        number_points_each_dimension=number_points_each_dimension, distribution=distribution,
        parallel=parallel, thinning=thinning, n_burning=n_burning, max_steps_out=max_steps_out,
        n_iterations=n_iterations, same_correlation=same_correlation, debug=debug,
        number_points_each_dimension_debug=number_points_each_dimension_debug,
        monte_carlo_sbo=monte_carlo_sbo, n_samples_mc=n_samples_mc, n_restarts_mc=n_restarts_mc,
        factr_mc=factr_mc, maxiter_mc=maxiter_mc, n_restarts=n_restarts,
        n_best_restarts=n_best_restarts, use_only_training_points=use_only_training_points,
        method_optimization=method_optimization, n_samples_parameters=n_samples_parameters,
        n_restarts_mean=n_restarts_mean, n_best_restarts_mean=n_best_restarts_mean,
        n_best_restarts_mc=n_best_restarts_mc)


    print ujson.dumps(spec, indent=4)

