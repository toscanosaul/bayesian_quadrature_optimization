import ujson

from scipy.stats import gamma
import numpy as np
from scipy.stats import poisson
import json
import os

from problems.vendor_problem.vendor import p1,p2,p3,p4,p5
from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    SBO_METHOD,
    EI_METHOD,
    DOGLEG,
    MULTI_TASK_METHOD,
    EXPONENTIAL,
    SCALED_KERNEL,
    GAMMA,
    WEIGHTED_UNIFORM_FINITE,
    SDE_METHOD,
    LBFGS_NAME,
)

if __name__ == '__main__':
    #simplex_domain = [1]
    simplex_domain = None
    # usage: python -m scripts.generate_multiple_spec > data/multiple_specs/multiple_test_spec.json

    # script used to generate spec file to run BGO
    # ARXIV
    # dim_x = [4]
    # bounds_domain_x = [[(0.01, 5.0), (0.0, 2.1), (1, 21), (1, 201)]]
    # problem_name = ['arxiv']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[5, 4, 5]]
    # bounds_domain = [[[0.01, 5.0], [0.0, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]]
    # #old bounds: [[0.01, 1.01], [0.01, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    # n_training = [5]
    # random_seed = range(1,101)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 1]]
    # x_domain = [[0, 1, 2, 3]]
    # number_points_each_dimension = [[6, 6, 11, 6]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [100]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SBO_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [DOGLEG]
    # maxepoch = [10]
    # n_samples_parameters_mean = [20]
    # maxepoch_mean = [20]
    # threshold_sbo = [0.001]
    # parallel_training = [True]

#    arxiv ei
#     dim_x = [4]
#     bounds_domain_x = [[(0.01, 5.0), (0.0, 2.1), (1, 21), (1, 201)]]
#     problem_name = ['arxiv_ei']
#     training_name = [None]
#     type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
#     dimensions = [[4]]
#     bounds_domain = [[[0.01, 5.0], [0.0, 2.1], [1, 21], [1, 201]]]
#     n_training = [1]
#     random_seed = range(1,101)
#     n_specs = len(random_seed)
#     type_bounds = [[0, 0, 0, 0]]
#     x_domain = [[0, 1, 2, 3]]
#     number_points_each_dimension = [[6, 6, 11, 6]]
#     mle = [False]
#     distribution = [UNIFORM_FINITE]
#     parallel = [True]
#     thinning = [10]
#     n_burning = [500]
#     max_steps_out = [1000]
#     n_iterations = [20]
#     same_correlation = [True]
#     debug = [False]
#     number_points_each_dimension_debug = [[10, 10, 10, 10]]
#     monte_carlo_sbo = [True]
#     n_samples_mc = [2]
#     n_restarts_mc = [5]
#     n_best_restarts_mc = [0]
#     factr_mc = [1e12]
#     maxiter_mc = [10]
#     n_restarts = [10]
#     n_best_restarts = [0]
#     use_only_training_points = [True]
#     method_optimization = [EI_METHOD]
#     n_samples_parameters = [5]
#     n_restarts_mean = [100]
#     n_best_restarts_mean = [10]
#     method_opt_mc = [DOGLEG]
#     maxepoch = [10]
#     n_samples_parameters_mean = [20]
#     maxepoch_mean = [20]
#     threshold_sbo = [0.001]
#     parallel_training = [False]
#     noises = [False]
#     n_sampless = [0]
#     parameters_distributions = [None]


    # #cnn
    # dim_x = [6]
    # bounds_domain_x = [[(1, 11), (4, 500), (0, 1), (3, 10), (100, 1000), (2, 6)]]
    # problem_name = ['cnn_cifar10']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[7, 6, 5]]
    # bounds_domain = [[[1, 11], [4, 500], [0, 1], [3, 10], [100, 1000], [2, 6],
    #                   [0, 1, 2, 3, 4]]]
    # #old bounds: [[0.01, 1.01], [0.01, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    # n_training = [10]
    # random_seed = range(1, 200)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 1]]
    # x_domain = [[0, 1, 2, 3, 4, 5]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SBO_METHOD]
    # # change to 2 if want to run again
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [None]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]


    # #multi_Task_cnn
    # dim_x = [6]
    # bounds_domain_x = [[(1, 11), (4, 500), (0, 1), (3, 10), (100, 1000), (2, 6)]]
    # problem_name = ['cnn_cifar10']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[7, 6, 5]]
    # bounds_domain = [[[1, 11], [4, 500], [0, 1], [3, 10], [100, 1000], [2, 6],
    #                   [0, 1, 2, 3, 4]]]
    # #old bounds: [[0.01, 1.01], [0.01, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    # n_training = [10]
    # random_seed = range(1, 200)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 1]]
    # x_domain = [[0, 1, 2, 3, 4, 5]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [100]
    # n_best_restarts = [10]
    # use_only_training_points = [True]
    # method_optimization = [MULTI_TASK_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [None]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]

    #EI_cnn
    # dim_x = [6]
    # bounds_domain_x = [[(1, 11), (4, 500), (0, 1), (3, 10), (100, 1000), (2, 6)]]
    # problem_name = ['cnn_cifar10_ei']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    # dimensions = [[6]]
    # bounds_domain = [[[1, 11], [4, 500], [0, 1], [3, 10], [100, 1000], [2, 6]]]
    # #old bounds: [[0.01, 1.01], [0.01, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    # n_training = [2]
    # random_seed = range(1, 100)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0]]
    # x_domain = [[0, 1, 2, 3, 4, 5]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [7]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [100]
    # n_best_restarts = [10]
    # use_only_training_points = [True]
    # method_optimization = [EI_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [None]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]

    ######### inventory
    # #### SBO
    customers = 10
    N = customers

    log_factorials = {}
    log_factorials[0] = 0
    log_factorials[1] = 0
    for i in range(2, N + 1):
        log_factorials[i] = np.log(i) + log_factorials[i - 1]

    weights = []
    domain_random = []
    for n1 in range(N + 1):
        for n2 in range(N + 1 - n1):
            for n3 in range(N + 1 - n1 - n2):
                for n4 in range(N + 1 - n1 - n2 - n3):
                    if n1 + n2 + n3 + n4 > N:
                        continue
                    n5 = N - n1 - n2 - n3 - n4
                    domain_random.append([n1, n2, n3, n4])
                    l_prob = n1 * np.log(p1) + n2 * np.log(p2) + n3 * np.log(p3) + n4 * np.log(
                        p4) + n5 * np.log(p5)
                    term_2 = log_factorials[N] - log_factorials[n1] - log_factorials[n2] - \
                             log_factorials[n3] - log_factorials[n4] - log_factorials[n5]
                    weights.append(np.exp(l_prob + term_2))

    simplex_domain = [customers]
    dim_x = [2]
    bounds_domain_x = [[(0, customers), (0, customers)]]
    problem_name = ['vendor_problem']
    training_name = [None]
    type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    dimensions = [[6]]
    bounds_domain = [[[0, customers], [0, customers], [0, customers], [0, customers], [0, customers], [0, customers]]]
    n_training = [4]
    random_seed = range(1, 500)
    n_specs = len(random_seed)
    type_bounds = [[0, 0, 0, 0]]
    x_domain = [[0, 1]]
    number_points_each_dimension = [[6, 6]]
    mle = [False]
    distribution = [WEIGHTED_UNIFORM_FINITE]
    parallel = [True]
    thinning = [10]
    n_burning = [500]
    max_steps_out = [1000]
    n_iterations = [35]
    same_correlation = [True]
    debug = [False]
    number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    monte_carlo_sbo = [True]
    n_samples_mc = [5]
    n_restarts_mc = [5]
    n_best_restarts_mc = [0]
    factr_mc = [1e12]
    maxiter_mc = [10]
    n_restarts = [10]
    n_best_restarts = [0]
    use_only_training_points = [True]
    method_optimization = [SBO_METHOD]
    n_samples_parameters = [5]
    n_restarts_mean = [100]
    n_best_restarts_mean = [10]
    method_opt_mc = [LBFGS_NAME]
    maxepoch = [50]
    n_samples_parameters_mean = [5]
    maxepoch_mean = [50]
    threshold_sbo = [0.001]
    parallel_training = [False]
    noises = [True]
    n_sampless = [3]

    parameters_distributions = [{'weights': weights,
                                  'domain_random': domain_random,
                                 'n_samples':10}]

    # #### multi_task
    # dim_x = [2]
    # bounds_domain_x = [[(0, customers), (0, customers)]]
    # problem_name = ['vendor_problem_multi_task']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[3, 2, 4]]
    # bounds_domain = [[[0, customers], [0, customers], [0, 1, 2, 3]]]
    # n_training = [4]
    # random_seed = range(1, 100)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 1]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [100]
    # n_best_restarts = [10]
    # use_only_training_points = [True]
    # method_optimization = [MULTI_TASK_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [DOGLEG]
    # maxepoch = [10]
    # n_samples_parameters_mean = [20]
    # maxepoch_mean = [20]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [True]
    # n_sampless = [5]
    # parameters_distributions = None

    ##EI
    customers = 1000
    #
    # dim_x = [2]
    # bounds_domain_x = [[(0, customers), (0, customers)]]
    # problem_name = ['vendor_problem_ei']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    # dimensions = [[2]]
    # bounds_domain = [[[0, customers], [0, customers]]]
    # n_training = [4]
    # random_seed = range(1, 501)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [GAMMA]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [EI_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [DOGLEG]
    # maxepoch = [10]
    # n_samples_parameters_mean = [20]
    # maxepoch_mean = [20]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [True]
    # n_sampless = [3]
    # parameters_distributions = [{'scale': [1.0], 'a': [customers]}]

    #BRANIN-SBO

    # domain_random = [[0.25, 0.2], [0.25, 0.4], [0.25, 0.6], [0.25, 0.8],
    #                           [0.5, 0.2], [0.5, 0.4], [0.5, 0.6], [0.5, 0.8],
    #                           [0.75, 0.2], [0.75, 0.4], [0.75, 0.6], [0.75, 0.8]]
    #
    # dim_x = [2]
    # bounds_domain_x = [[(0, 1), (0, 1)]]
    # problem_name = ['branin']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    # dimensions = [[4]]
    # bounds_domain = [[[0, 1], [0, 1], [0.25, 0.5, 0.75], [0.2, 0.4, 0.6, 0.8]]]
    # n_training = [12]
    # random_seed = range(1, 201)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 1, 1]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [42]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SBO_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [{'weights': [0.0375, 0.0875, 0.0875, 0.0375, 0.0750, 0.1750,
    #                                          0.1750, 0.0750, 0.0375, 0.0875, 0.0875, 0.0375],
    #                              'domain_random': domain_random}]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]
    #
    # ##branin- EI
    # dim_x = [2]
    # bounds_domain_x = [[(0, 1), (0, 1)]]
    # problem_name = ['branin_ei']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    # dimensions = [[2]]
    # bounds_domain = [[(0, 1), (0, 1)]]
    # n_training = [12]
    # random_seed = range(1, 10)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [100]
    # max_steps_out = [1000]
    # n_iterations = [15]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [1]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [EI_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [1]
    # n_best_restarts_mean = [0]
    # method_opt_mc = [DOGLEG]
    # maxepoch = [10]
    # n_samples_parameters_mean = [10]
    # maxepoch_mean = [10]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [True]
    # n_sampless = [5]
    # parameters_distributions = None
    #





    #branin-sde
    # domain_random = [[0.25, 0.2], [0.25, 0.4], [0.25, 0.6], [0.25, 0.8],
    #                           [0.5, 0.2], [0.5, 0.4], [0.5, 0.6], [0.5, 0.8],
    #                           [0.75, 0.2], [0.75, 0.4], [0.75, 0.6], [0.75, 0.8]]
    #
    # dim_x = [2]
    # bounds_domain_x = [[(0, 1), (0, 1)]]
    # problem_name = ['branin']
    # training_name = [None]
    # type_kernel = [[MATERN52_NAME]]
    # dimensions = [[4]]
    # bounds_domain = [[[0, 1], [0, 1], [0.25, 0.5, 0.75], [0.2, 0.4, 0.6, 0.8]]]
    # n_training = [12]
    # random_seed = range(1, 701)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 1, 1]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [42]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [50]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SDE_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean  = [10]
    # n_best_restarts_mean = [0]
    # method_opt_mc = [DOGLEG]
    # maxepoch = [10]
    # n_samples_parameters_mean = [20]
    # maxepoch_mean = [20]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [{'weights': [0.0375, 0.0875, 0.0875, 0.0375, 0.0750, 0.1750,
    #                                          0.1750, 0.0750, 0.0375, 0.0875, 0.0875, 0.0375],
    #                              'domain_random': domain_random}]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]

    #multi_task branin

    # dim_x = [2]
    # bounds_domain_x = [[(0, 1), (0, 1)]]
    # problem_name = ['branin_mt']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[3, 2, 12]]
    # bounds_domain = [[[0, 1], [0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]
    # n_training = [12]
    # random_seed = range(1, 501)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 1]]
    # x_domain = [[0, 1]]
    # number_points_each_dimension = [[6, 6]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [46]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [100]
    # n_best_restarts = [10]
    # use_only_training_points = [True]
    # method_optimization = [MULTI_TASK_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # domain_random = [[0], [1], [2],[3],[4],[5],[6],[7],[8],[9],[10],[11]]
    # parameters_distributions = [{'weights': [0.0375, 0.0875, 0.0875, 0.0375, 0.0750, 0.1750,
    #                                          0.1750, 0.0750, 0.0375, 0.0875, 0.0875, 0.0375],
    #                              'domain_random': domain_random}]
    #
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]


    ##citibike_mt
    ############################### Parameters needed
    # n1 = 4
    # n2 = 1
    #
    # nDays = 365
    #
    # nSets = 4
    #
    # fil = "poissonDays.txt"
    # fil = "problems/citi_bike_mt/" + fil
    # poissonParameters = np.loadtxt(fil)
    #
    # ###readData
    #
    # poissonArray = [[] for i in xrange(nDays)]
    # exponentialTimes = [[] for i in xrange(nDays)]
    #
    # for i in xrange(nDays):
    #     fil = "daySparse" + "%d" % i + "ExponentialTimesNonHom.txt"
    #     fil2 = os.path.join("problems/citi_bike_mt/SparseNonHomogeneousPP2", fil)
    #     poissonArray[i].append(np.loadtxt(fil2))
    #
    #     fil = "daySparse" + "%d" % i + "PoissonParametersNonHom.txt"
    #     fil2 = os.path.join("problems/citi_bike_mt/SparseNonHomogeneousPP2", fil)
    #     exponentialTimes[i].append(np.loadtxt(fil2))
    #
    # numberStations = 329
    # Avertices = [[]]
    # for j in range(numberStations):
    #     for k in range(numberStations):
    #         Avertices[0].append((j, k))
    #
    # with open('problems/citi_bike_mt/json.json') as data_file:
    #     data = json.load(data_file)
    #
    # f = open('problems/citi_bike_mt/' + str(4) + "-cluster.txt", 'r')
    # cluster = eval(f.read())
    # f.close()
    #
    # bikeData = np.loadtxt("problems/citi_bike_mt/bikesStationsOrdinalIDnumberDocks.txt", skiprows=1)
    #
    # TimeHours = 4.0
    # numberBikes = 6000
    #
    # poissonParameters *= TimeHours
    #
    # ###upper bounds for X
    # upperX = np.zeros(n1)
    # temBikes = bikeData[:, 2]
    # for i in xrange(n1):
    #     temp = cluster[i]
    #     indsTemp = np.array([a[0] for a in temp])
    #     upperX[i] = np.sum(temBikes[indsTemp])
    #
    #
    # ##weights of w
    # def computeProbability(w, parLambda, nDays):
    #     probs = poisson.pmf(w, mu=np.array(parLambda))
    #     probs *= (1.0 / nDays)
    #     return np.sum(probs)
    #
    #
    # L = 3829
    # M = 4010
    # wTemp = np.array(range(L, M))
    # probsTemp = np.zeros(M - L)
    # for i in range(M - L):
    #     probsTemp[i] = computeProbability(wTemp[i], poissonParameters, nDays)
    #
    # n_tasks = M - L
    # ###################
    # dim_x = [3]
    # bounds_domain_x = [[(0, upperX[0]), (0, upperX[1]), (0, upperX[2])]]
    # simplex_domain = [numberBikes]
    #
    # problem_name = ['citi_bike_mt']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[4, 3, n_tasks]]
    # bounds_domain = [[[0, upperX[0]], [0, upperX[1]], [0, upperX[2]], range(n_tasks)]]
    # n_training = [50]
    # random_seed = range(1, 501)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 1]]
    # x_domain = [[0, 1, 2]]
    # number_points_each_dimension = [[6, 6, 6]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [21]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [100]
    # n_best_restarts = [10]
    # use_only_training_points = [True]
    # method_optimization = [MULTI_TASK_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # n_sampless = [5]
    # domain_random = [[i] for i in range(n_tasks)]
    # parameters_distributions = [{'weights': list(probsTemp),
    #                              'domain_random': domain_random}]
    #
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]
    # noises = [True]



    ##mnist-kg
    # dim_x = [5]
    # bounds_domain_x = [[(0.2, 0.99), (0.0, 1), (100, 5000), (0, 1), (1, 200)]]
    # problem_name = ['mnist_logistic_kg']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    # dimensions = [[6, 5, 1]]
    # bounds_domain = [[[0.2, 0.99], [0.0, 1], [100, 5000], [0, 1], [1, 200], [0]]]
    # #old bounds: [[0.01, 1.01], [0.01, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
    # n_training = [5]
    # random_seed = range(1, 200)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 1]]
    # x_domain = [[0, 1, 2, 3, 4]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SBO_METHOD]
    # # change to 2 if want to run again
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # #I should have used 50 instead of 20
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # parameters_distributions = [None]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]


    #aircraft
    #sbo
    # dim_x = [8]
    #
    # heights = [40000, 50000, 30000, 10000, 20000, 7000]
    # mach_numbers =  [0.8322926591514808,  0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
    #                  0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
    #
    # bounds_domain_x = [[(0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (10.0, 15.0), (10.0, 15.0),
    #                     (10.0, 15.0), (10.0, 15.0), (10.0, 15.0)]]
    # domain_random = [[m, h] for m in mach_numbers for h in heights]
    #
    # problem_name = ['aircraft']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    # dimensions = [[10]]
    # bounds_domain = [[[0.01, 0.5], [0.01, 0.5], [0.01, 0.5], [10.0, 15.0], [10.0, 15.0], [10.0, 15.0],
    #                   [10.0, 15.0], [10.0, 15.0], mach_numbers, heights]]
    # n_training = [3]
    # random_seed = range(1, 1001)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    # x_domain = [[0, 1, 2,3,4,5,6,7]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1, 1, 1]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SBO_METHOD]
    #
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # n_points = len(domain_random)
    # parameters_distributions = [{'weights': n_points * [1.0 / float(n_points)],
    #                               'domain_random': domain_random}]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]
    #
    # #EI
    # #aircraft- EI
    # dim_x = [8]
    #
    # heights = [40000, 50000, 30000, 10000, 20000, 7000]
    # mach_numbers =  [0.8322926591514808,  0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
    #                  0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
    #
    # bounds_domain_x = [[(0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (10.0, 15.0), (10.0, 15.0),
    #                     (10.0, 15.0), (10.0, 15.0), (10.0, 15.0)]]
    # domain_random = [[m, h] for m in mach_numbers for h in heights]
    #
    # problem_name = ['aircraft_ei']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    #
    # dimensions = [[8]]
    # bounds_domain = [[[0.01, 0.5], [0.01, 0.5], [0.01, 0.5], [10.0, 15.0], [10.0, 15.0], [10.0, 15.0],
    #                   [10.0, 15.0], [10.0, 15.0]]]
    #
    # n_training = [1]
    # random_seed = range(1, 100)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 0, 0]]
    # x_domain = [[0, 1, 2, 3, 4, 5, 6, 7]]
    #
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1, 1, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    #
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [15]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [EI_METHOD]
    #
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # n_points = len(domain_random)
    # parameters_distributions = [{'weights': n_points * [1.0 / float(n_points)],
    #                               'domain_random': domain_random}]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]
    #
    #
    # ### aircraft_multi_task
    #
    # dim_x = [8]
    #
    # n_tasks = 48
    #
    # heights = [40000, 50000, 30000, 10000, 20000, 7000]
    # mach_numbers =  [0.8322926591514808,  0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
    #                  0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
    #
    # bounds_domain_x = [[(0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (10.0, 15.0), (10.0, 15.0),
    #                     (10.0, 15.0), (10.0, 15.0), (10.0, 15.0)]]
    # problem_name = ['aircraft_mt']
    # training_name = [None]
    # type_kernel = [[PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]]
    #
    # dimensions = [[9, 8, n_tasks]]
    # bounds_domain = [[[0.01, 0.5], [0.01, 0.5], [0.01, 0.5], [10.0, 15.0], [10.0, 15.0], [10.0, 15.0],
    #                   [10.0, 15.0], [10.0, 15.0], range(n_tasks)]]
    # n_training = [3]
    # random_seed = range(1, 1001)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 0, 0, 1]]
    # x_domain = [[0, 1, 2,3,4,5,6,7]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1, 1, 1]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [35]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [MULTI_TASK_METHOD]
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # n_sampless = [0]
    # domain_random = [[i] for i in range(n_tasks)]
    # n_points = len(domain_random)
    # parameters_distributions = [{'weights': n_points * [1.0 / float(n_points)],
    #                              'domain_random': domain_random}]
    #
    #
    # ### aircraft_sde
    # ## SDE NEEDS AT LEAST 4 TRAINING EXAMPLES!!!!!
    #
    # dim_x = [8]
    #
    # heights = [40000, 50000, 30000, 10000, 20000, 7000]
    # mach_numbers =  [0.8322926591514808,  0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
    #                  0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
    #
    # bounds_domain_x = [[(0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (10.0, 15.0), (10.0, 15.0),
    #                     (10.0, 15.0), (10.0, 15.0), (10.0, 15.0)]]
    # domain_random = [[m, h] for m in mach_numbers for h in heights]
    #
    # problem_name = ['aircraft']
    # training_name = [None]
    # type_kernel = [[MATERN52_NAME]]
    # dimensions = [[10]]
    # bounds_domain = [[[0.01, 0.5], [0.01, 0.5], [0.01, 0.5], [10.0, 15.0], [10.0, 15.0], [10.0, 15.0],
    #                   [10.0, 15.0], [10.0, 15.0], mach_numbers, heights]]
    # n_training = [4]
    # random_seed = range(1, 1001)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    # x_domain = [[0, 1, 2,3,4,5,6,7]]
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1, 1, 1]]
    # mle = [False]
    # distribution = [WEIGHTED_UNIFORM_FINITE]
    # parallel = [True]
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [25]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [SDE_METHOD]
    #
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [False]
    # n_sampless = [0]
    # n_points = len(domain_random)
    # parameters_distributions = [{'weights': n_points * [1.0 / float(n_points)],
    #                               'domain_random': domain_random}]
    # optimize_only_posterior_means = [False]
    # start_optimize_posterior_means = [0]
    #
    # #noisy-ei
    #
    # dim_x = [8]
    #
    # heights = [40000, 50000, 30000, 10000, 20000, 7000]
    # mach_numbers =  [0.8322926591514808,  0.8338867889125835, 0.8366730174675625, 0.8389325581307193,
    #                  0.8410674418692806, 0.8433269825324374, 0.8461132110874164, 0.8477073408485192]
    #
    # bounds_domain_x = [[(0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (10.0, 15.0), (10.0, 15.0),
    #                     (10.0, 15.0), (10.0, 15.0), (10.0, 15.0)]]
    # domain_random = [[m, h] for m in mach_numbers for h in heights]
    #
    # problem_name = ['noisy_aircraft_ei']
    # training_name = [None]
    # type_kernel = [[SCALED_KERNEL, MATERN52_NAME]]
    #
    # dimensions = [[8]]
    # bounds_domain = [[[0.01, 0.5], [0.01, 0.5], [0.01, 0.5], [10.0, 15.0], [10.0, 15.0], [10.0, 15.0],
    #                   [10.0, 15.0], [10.0, 15.0]]]
    #
    # n_training = [1]
    # random_seed = range(1, 2000)
    # n_specs = len(random_seed)
    # type_bounds = [[0, 0, 0, 0, 0, 0, 0, 0]]
    # x_domain = [[0, 1, 2, 3, 4, 5, 6, 7]]
    #
    # number_points_each_dimension = [[6, 6, 11, 6, 1, 1, 1, 1]]
    # mle = [False]
    # distribution = [UNIFORM_FINITE]
    # parallel = [True]
    #
    # thinning = [10]
    # n_burning = [500]
    # max_steps_out = [1000]
    # n_iterations = [15]
    # same_correlation = [True]
    # debug = [False]
    # number_points_each_dimension_debug = [[10, 10, 10, 10, 10, 10]]
    # monte_carlo_sbo = [True]
    # n_samples_mc = [5]
    # n_restarts_mc = [5]
    # n_best_restarts_mc = [0]
    # factr_mc = [1e12]
    # maxiter_mc = [10]
    # n_restarts = [10]
    # n_best_restarts = [0]
    # use_only_training_points = [True]
    # method_optimization = [EI_METHOD]
    #
    # n_samples_parameters = [5]
    # n_restarts_mean = [100]
    # n_best_restarts_mean = [10]
    # method_opt_mc = [LBFGS_NAME]
    # maxepoch = [50]
    # n_samples_parameters_mean = [5]
    # maxepoch_mean = [50]
    # threshold_sbo = [0.001]
    # parallel_training = [False]
    # noises = [True]
    # n_sampless = [5]
    # n_points = len(domain_random)
    # parameters_distributions = [{'weights': n_points * [1.0 / float(n_points)],
    #                               'domain_random': domain_random}]
    optimize_only_posterior_means = [False]
    start_optimize_posterior_means = [0]



    specs = SpecService.generate_dict_multiple_spec(
        n_specs=n_specs, problem_names=problem_name, dim_xs=dim_x, bounds_domain_xs=bounds_domain_x,
        training_names=training_name, type_kernels=type_kernel, dimensionss=dimensions,
        bounds_domains=bounds_domain, number_points_each_dimensions=number_points_each_dimension,
        n_trainings=n_training, random_seeds=random_seed, type_boundss=type_bounds,
        x_domains=x_domain, mles=mle, distributions=distribution, parallels=parallel,
        thinnings=thinning, n_burnings=n_burning, max_steps_outs=max_steps_out,
        n_iterationss=n_iterations, same_correlations=same_correlation, debugs=debug,
        number_points_each_dimension_debugs=number_points_each_dimension_debug,
        monte_carlo_sbos=monte_carlo_sbo, n_samples_mcs=n_samples_mc, n_restarts_mcs=n_restarts_mc,
        factr_mcs=factr_mc, maxiter_mcs=maxiter_mc, n_restartss=n_restarts,
        n_best_restartss=n_best_restarts, use_only_training_pointss=use_only_training_points,
        method_optimizations=method_optimization, n_samples_parameterss=n_samples_parameters,
        n_restarts_means=n_restarts_mean, n_best_restarts_means=n_best_restarts_mean,
        n_best_restarts_mcs=n_best_restarts_mc, maxepochs=maxepoch, method_opt_mcs=method_opt_mc,
        n_samples_parameters_means=n_samples_parameters_mean, maxepoch_means=maxepoch_mean,
        threshold_sbos=threshold_sbo, parallel_trainings=parallel_training, noises=noises,
        n_sampless=n_sampless, parameters_distributions=parameters_distributions,
        optimize_only_posterior_means=optimize_only_posterior_means,
        start_optimize_posterior_means=start_optimize_posterior_means,
        simplex_domain=simplex_domain
    )

    print ujson.dumps(specs, indent=4)
