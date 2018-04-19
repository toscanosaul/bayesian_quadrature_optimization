from __future__ import absolute_import

import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt

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
    ORNSTEIN_KERNEL,
)
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.lib.la_functions import (
    cholesky,
    cho_solve,
)
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)



class StatModel(object):

    def __init__(self, raw_results, best_result, current_iteration, get_value_next_iteration,
                 starting_point, current_batch_index, problem_name=None, max_iterations=1000,
                 parametric_mean=False, square_root_factor=True, divide_kernel_prod_factor=True):
        """

        :param raw_results: [float]
        :param best_result: float
        :param current_iteration: int

        """
        self.starting_point = starting_point
        self.current_point = starting_point
        self.current_batch_index = current_batch_index

        self.raw_results = raw_results
        self.best_result = best_result
        self.current_iteration = current_iteration

        self.differences = None
        self.points_differences = None
        self.training_data = None
        self.process_data()

        self.problem_name = problem_name
        self.max_iterations = max_iterations
        self.parametric_mean = parametric_mean
        self.gp_model = None
        self.define_gp_model()

        self.function_factor_kernel = lambda x: float(x)
        if square_root_factor:
            self.function_factor_kernel = lambda x: np.sqrt(x)

        self.divisor_kernel = lambda x, y: max(x, y)
        if divide_kernel_prod_factor:
            self.divisor_kernel = lambda x, y: x * y

        self.set_samplers()

        self.get_value_next_iteration = get_value_next_iteration

    def process_data(self):
        differences = []
        for i in range(1, len(self.raw_results)):
            diff = self.raw_results[i] - self.raw_results[i-1]
            differences.append(diff)
        self.differences = differences

        points_diff = []
        for i in range(1, len(self.raw_results)):
            points_diff.append((i, i + 1))
        self.points_differences = points_diff

        training_data = {}
        training_data['points'] = [[float(i)] for i in range(2, len(self.differences) + 2)]
        training_data['evaluations'] = list(self.differences)
        training_data['var_noise'] = []
        self.training_data = training_data

    def define_gp_model(self):

        def toy_objective_function(x):
            return [np.sum(x)]

        spec = {
            'name_model': 'gp_fitting_gaussian',
            'problem_name': self.problem_name,
            'type_kernel': [ORNSTEIN_KERNEL],
            'dimensions': [1],
            'bounds_domain': [[1, self.max_iterations]],
            'type_bounds': [0],
            'n_training': 10,
            'noise': False,
            'training_data': self.training_data,
            'points': None,
            'training_name': None,
            'mle': False,
            'thinning': 10,
            'n_burning': 500,
            'max_steps_out': 1000,
            'n_samples': 0,
            'random_seed': 1,
            'kernel_values': None,
            'mean_value': None,
            'var_noise_value': None,
            'cache': True,
            'same_correlation': True,
            'use_only_training_points': True,
            'optimization_method': 'SBO',
            'n_samples_parameters': 10,
            'parallel_training': False,
            'simplex_domain': None,
            'objective_function': toy_objective_function,
            'define_samplers': False
        }

        model = GPFittingService.from_dict(spec)
        model.dimension_parameters -= 2
        model.best_result = self.best_result
        model.current_iteration = self.current_iteration
        model.raw_results = list(self.raw_results)
        model.data['points'] = list(self.points_differences)
        model.mean_params = []

        self.gp_model = model
        self.set_samplers(self.gp_model)

    def covariance_diff_kernel(self, gp_model, x_poins, params):
        # f = lambda x: np.sqrt(float(x))
        #
        # if linear:
        #     f = lambda x: float(x)
        f = self.function_factor_kernel
        g = self.divisor_kernel
        raw_x = []
        raw_x.append([x_poins[0][0]])
        for t in x_poins:
            raw_x.append([t[1]])
        raw_x = np.array(raw_x)
        raw_cov = gp_model.kernel.evaluate_cov_defined_by_params(params, raw_x, 1)
        n = len(raw_x)
        cov_mat = np.zeros((n, n))
        for i in range(n):
            diff = (raw_cov[i, i] / g(f(i+1), f(i+1))) \
                   + (raw_cov[i + 1, i + 1] / g(f(i+2),f(i+2)))

            diff -= (2.0 * raw_cov[i, i + 1] / (g(f(i + 2), f(i+1))))
            cov_mat[i, i] = diff
            for j in range(i + 1, n):
                diff = (raw_cov[i, j] / (g(f(j + 1), f(i + 1)))) +\
                       (raw_cov[i + 1, j + 1] / (g(f(j + 2), f(i + 2))))
                diff -= ((raw_cov[i, j + 1] / (g(f(j + 2), f(i + 1)))) +
                         (raw_cov[i + 1, j] / (g(f(j + 1), f(i + 2)))))
                cov_mat[i, j] = diff
                cov_mat[j, i] = diff
        return cov_mat


    def log_likelihood(self, gp_model, parameters_kernel, mean_parameters=None):
        """
        GP log likelihood: y(x) ~ f(x) + epsilon, where epsilon(x) are iid N(0,var_noise), and
        f(x) ~ GP(mean, cov)

        :param var_noise: (float) variance of the noise
        :param mean: (float)
        :param parameters_kernel: np.array(k), The order of the parameters is given in the
            definition of the class kernel.
        :return: float

        """

        X_data = gp_model.data['points']
        cov = self.covariance_diff_kernel(gp_model, X_data, parameters_kernel)
        chol = cholesky(cov, max_tries=7)

        if self.parametric_mean:
            mean = self.compute_parametric_mean(mean_parameters)
        else:
            mean = 0
        y_unbiased = gp_model.data['evaluations'] - mean
        solve = cho_solve(chol, y_unbiased)

        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(y_unbiased, solve)

    def compute_parametric_mean(self, mean_parameters):
        pass

    def compute_prior_mean(self, mean_parameters):
        pass

    def log_prob(self, parameters, gp_model, mean_parameters=None):
        lp = 0.0
        parameters_model = gp_model.get_parameters_model[2:]
        index = 0

        for parameter in parameters_model:
            dimension = parameter.dimension
            lp += parameter.log_prior(parameters[index: index + dimension])
            index += dimension

        if not np.isinf(lp) and self.parametric_mean:
            lp += self.compute_prior_mean(mean_parameters)

        if not np.isinf(lp):
            lp += self.log_likelihood(gp_model, parameters, mean_parameters)

        return lp

    def sample_parameters(self, gp_model, n_samples, start_point=None, random_seed=None):
        """
        Sample parameters of the model from the posterior without considering burning.

        :param n_samples: (int)
        :param start_point: np.array(n_parameters)
        :param random_seed: int

        :return: n_samples * [np.array(float)]
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        samples = []

        if start_point is None:
            start_point = gp_model.samples_parameters[-1]


        n_samples *= (gp_model.thinning + 1)
        n_samples = int(n_samples)

        if len(gp_model.slice_samplers) == 1:
            for sample in range(n_samples):
                start_point_ = None
                n_try = 0
                points = start_point
                while start_point_ is None and n_try < 10:
                    try:
                        start_point_ = \
                            gp_model.slice_samplers[0].slice_sample(points, None, *(gp_model, ))
                    except Exception as e:
                        n_try += 1
                        start_point_ = None
                if start_point_ is None:
                    logger.info('program failed to compute a sample of the parameters')
                    sys.exit(1)
                start_point = start_point_
                samples.append(start_point)
            samples_return = samples[::gp_model.thinning + 1]
            gp_model.samples_parameters += samples_return
            return samples_return

        return []

    def set_samplers(self, gp_model):
        """
        Defines the samplers of the parameters of the model.
        We assume that we only have one set of length scale parameters.
        """
        gp_model.slice_samplers = []
        gp_model.samples_parameters = []
        gp_model.start_point_sampler = []

        ignore_index = None

        if not self.parametric_mean:
            slice_parameters = {
                'max_steps_out': gp_model.max_steps_out,
                'component_wise': True,
            }
        else:
            slice_parameters = {
                'max_steps_out': gp_model.max_steps_out,
                'component_wise': False,
            }

        gp_model.slice_samplers.append(SliceSampling(
            self.log_prob, range(gp_model.dimension_parameters), ignore_index=ignore_index,
            **slice_parameters))

        if gp_model.start_point_sampler is not None and len(gp_model.start_point_sampler) > 0:
            if len(gp_model.samples_parameters) == 0:
                gp_model.samples_parameters.append(np.array(gp_model.start_point_sampler))
        else:
            gp_model.samples_parameters = []
            z = gp_model.get_value_parameters_model
            z = z[2:]

            if self.parametric_mean:
                mean_params = gp_model.mean_params
                gp_model.samples_parameters.append(np.concatenate((z, mean_params)))
            else:
                gp_model.samples_parameters.append(z)

            if gp_model.n_burning > 0:
                parameters = self.sample_parameters(
                    gp_model, float(gp_model.n_burning) / (gp_model.thinning + 1))
                gp_model.samples_parameters = []
                gp_model.samples_parameters.append(parameters[-1])
                gp_model.start_point_sampler = parameters[-1]
            else:
                gp_model.start_point_sampler = gp_model.samples_parameters[-1]


    def cov_diff_point(self, gp_model, kernel_params, x, X_hist):
        f = self.function_factor_kernel
        g = self.divisor_kernel

        z = np.array([x])

        raw_x = []
        raw_x.append([X_hist[0][0]])
        for t in X_hist:
            raw_x.append([t[1]])

        raw_x = np.array(raw_x)
        raw_cov = gp_model.kernel.evaluate_cross_cov_defined_by_params(kernel_params, z, raw_x, 1)

        cov = np.zeros(len(X_hist))
        for i in range(len(X_hist)):
            cov[i] = (raw_cov[0, i] / g(f(x[0]),f(i+1)))
            cov[i] -= raw_cov[0, i+1] / g(f(x[0]), f(i+2))
        return cov


    def compute_posterior_params(self, gp_model, kernel_params, mean_parameters=None):
        ## Mean is zero. We may need to change when including mean
        ##Compute posterior parameters of f(x*)

        f = self.function_factor_kernel
        g = self.divisor_kernel

        current_point = [gp_model.current_iteration]

        X_data = gp_model.data['points']
        vector_ = self.cov_diff_point(gp_model, kernel_params, current_point, X_data)

        cov = self.covariance_diff_kernel(gp_model, X_data, kernel_params)
        chol = cholesky(cov, max_tries=7)

        if self.parametric_mean:
            mean = self.compute_parametric_mean(mean_parameters)
        else:
            mean = 0.

        y_unbiased = gp_model.data['evaluations'] - mean
        solve = cho_solve(chol, y_unbiased)

        part_2 = cho_solve(chol, vector_)

        mean = gp_model.raw_results[-1] + np.dot(part_2, solve)

        raw_cov = gp_model.kernel.evaluate_cov_defined_by_params(
            kernel_params, np.array([current_point]), 1) / \
                  g(f(current_point[0]), f(current_point[0]))

        var = raw_cov - np.dot(part_2, part_2)

        return mean, var

    def compute_posterior_params_marginalize(self, gp_model, n_samples=10, burning_parameters=True):
        if burning_parameters:
            parameters = self.sample_parameters(
                gp_model, float(gp_model.n_burning) / (gp_model.thinning + 1))
            gp_model.samples_parameters = []
            gp_model.samples_parameters.append(parameters[-1])
            gp_model.start_point_sampler = parameters[-1]

        parameters = self.sample_parameters(gp_model, n_samples)

        dim_kernel_params = gp_model.dimension_parameters

        means = []
        covs = []
        for param in parameters:
            kernel_params = param[0: dim_kernel_params]
            mean_params = None

            if self.parametric_mean:
                mean_params = param[dim_kernel_params:]

            mean, cov = self.compute_posterior_params(gp_model, kernel_params, mean_params)
            means.append(mean)
            covs.append(cov)

        mean = np.mean(means)
        std = np.sqrt(np.mean(covs))

        ci = [mean - 1.96 * std, mean + 1.96 * std]

        return mean, std, ci

    def add_observations(self, gp_model, point, y):
        gp_model.current_iteration = point
        previous_y = gp_model.raw_results[-1]
        gp_model.raw_results.append(y)
        gp_model.best_result = max(gp_model.best_result, y)
        gp_model.data['points'].append((point-1, point))
        gp_model.data['evaluations'] = np.concatenate(
            (gp_model.data['evaluations'],[(y - previous_y)]))


    def accuracy(self, gp_model, start=3, iterations=21, sufix=None):
        means = []
        cis = []

        mean, std, ci = self.compute_posterior_params_marginalize(gp_model)
        means.append(mean)
        cis.append(ci)

        for i in range(start, iterations):
            print (i)
            if len(gp_model.raw_results) < i + 1:
                self.add_observations(gp_model, i+1, self.get_value_next_iteration(i+1))
            mean, std, ci = self.compute_posterior_params_marginalize(gp_model)
            means.append(mean)
            cis.append(ci)

        accuracy_results = {}
        accuracy_results['means'] = means
        accuracy_results['ci'] = cis
        file_name = 'data/multi_start/accuracy_results'

        if self.problem_name is not None:
            file_name += '_' + self.problem_name

        if sufix is not None:
            file_name += '_' + sufix

        JSONFile.write(accuracy_results, file_name + '.json')

        return means, cis

    def plot_accuracy_results(self, means, cis, original_value, start=3, sufix=None):
        plt.figure()
        x_lim = len(means)
        points = range(start, x_lim + start)
        plt.plot(points, means, 'b', label='means')
        plt.plot(points, len(points) * [original_value], label='final value')
        plt.plot(points, [t[0] for t in cis],'g-', label='ci')
        plt.plot(points, [t[1] for t in cis],'g-', label='ci')

        plt.legend()
        plt.ylabel('Objective function')
        plt.xlabel('Iteration')

        file_name = 'data/multi_start/accuracy_plot'
        if self.problem_name is not None:
            file_name += '_' + self.problem_name

        if sufix is not None:
            file_name += '_' + sufix
        plt.savefig(file_name + '.pdf')

    def save_model(self, sufix=None):
        stat_model_dict = {}
        stat_model_dict['current_point'] = self.current_point
        stat_model_dict['starting_point'] = self.starting_point
        stat_model_dict['current_batch_index'] = self.current_batch_index
        stat_model_dict['best_result'] = self.gp_model.best_result
        stat_model_dict['current_iteration'] = self.gp_model.current_iteration
        stat_model_dict['raw_results'] = self.gp_model.raw_results

        file_name = 'data/multi_start/stat_model'

        if self.problem_name is not None:
            file_name += '_' + self.problem_name

        if sufix is not None:
            file_name += '_' + sufix

        JSONFile.write(stat_model_dict, file_name + '.json')

