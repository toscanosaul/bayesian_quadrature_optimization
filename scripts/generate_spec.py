import ujson

import numpy as np

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
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
    n_training = 30
    random_seed = 5
    type_bounds = [0, 0, 0, 0, 1]
    x_domain = [0, 1, 2, 3]
    number_points_each_dimension = [6, 6, 11, 6]
    mle = True
    distribution = UNIFORM_FINITE
    parallel = False
    thinning = 5
    n_burning = 100
    max_steps_out = 1000
    n_iterations = 100

    # var_noise_value = [1.00001785e-10]
    # mean_value = [-8.81677684e+00]
    # kernel_values = [1.07477776e+02, 9.62543469e+00, -6.71792551e+00, -7.39010376e-02]

    ## Generate training data
    # np.random.seed(5)
    # n_points = 100
    # points = np.linspace(0, 100, n_points)
    # points = points.reshape([n_points, 1])
    # tasks = np.random.randint(2, size=(n_points, 1))
    #
    # add = [10, -10]
    # kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
    # function = SampleFunctions.sample_from_gp(points, kernel)
    #
    # for i in xrange(n_points):
    #     function[0, i] += add[tasks[i, 0]]
    # points = np.concatenate((points, tasks), axis=1)
    #
    # function = function[0, :]
    #
    # points_ls = [list(points[i, :]) for i in xrange(n_points)]
    #
    # training_data_med = {
    #     'evaluations': list(function[0:5]),
    #     'points': points_ls[0:5],
    #     "var_noise": [],
    # }

    ######

    spec = SpecService.generate_dict_spec(problem_name, dim_x, bounds_domain_x, training_name,
                                          type_kernel, dimensions, bounds_domain=bounds_domain,
                                          n_training=n_training, random_seed=random_seed,
                                          type_bounds=type_bounds, x_domain=x_domain, mle=mle,
                                          number_points_each_dimension=number_points_each_dimension,
                                          distribution=distribution, parallel=parallel,
                                          thinning=thinning, n_burning=n_burning,
                                          max_steps_out=max_steps_out, n_iterations=n_iterations)

    print ujson.dumps(spec, indent=4)

