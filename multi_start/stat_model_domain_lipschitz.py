from __future__ import absolute_import

import numpy as np
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
from scipy.stats import norm

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
from multi_start.parametric_functions import ParametricFunctions

logger = SBOLog(__name__)



class StatModelLipschitz(object):

    def __init__(self, raw_results, best_result, current_iteration, get_value_next_iteration,
                 starting_point, current_batch_index, current_epoch,
                 kwargs_get_value_next_iteration=None,
                 problem_name=None, max_iterations=1000,
                 parametric_mean=False, square_root_factor=True, divide_kernel_prod_factor=True,
                 lower=None, upper=None, total_batches=10, n_burning=500, n_thinning=10,
                 lipschitz=None, type_model='lipschitz'):
        """

        :param raw_results: [float]
        :param best_result: float
        :param current_iteration: int

        """
        self.starting_point = starting_point
        self.current_point = starting_point
        self.current_batch_index = current_batch_index
        self.total_batches = total_batches
        self.current_epoch = current_epoch
        self.n_burning = n_burning
        self.n_thinning = n_thinning

        self.raw_results = raw_results #dictionary with points and values, and gradients
        self.best_result = best_result
        self.current_iteration = current_iteration

        self.lipschitz = lipschitz
        self.type_model = type_model

        self.differences = None
        self.points_differences = None
        self.training_data = None
        self.process_data()

        self.problem_name = problem_name
        self.max_iterations = max_iterations
        self.parametric_mean = parametric_mean

        self.parametrics = None
        if self.parametric_mean:
            self.parametrics = ParametricFunctions(max_iterations, lower, upper)

        self.gp_model = None
        self.define_gp_model()

        self.function_factor_kernel = lambda x: float(x)
        if square_root_factor:
            self.function_factor_kernel = lambda x: np.sqrt(x)

        self.divisor_kernel = lambda x, y: max(x, y)
        if divide_kernel_prod_factor:
            self.divisor_kernel = lambda x, y: x * y

        self.set_samplers(self.gp_model)

        self.get_value_next_iteration = get_value_next_iteration
        self.kwargs = {}
        if kwargs_get_value_next_iteration is not None:
            self.kwargs = kwargs_get_value_next_iteration

    def process_data(self):
        differences = []
        for i in range(1, len(self.raw_results)):
            diff = -np.sqrt(i) * (np.array(self.raw_results['points'][i]) - np.array(self.raw_results['points'][i-1]))
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
            'thinning': self.n_thinning,
            'n_burning': self.n_burning,
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
        model.raw_results = dict(self.raw_results)
        model.data['points'] = list(self.points_differences)
        model.mean_params = []

        self.gp_model = model

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
        n = len(x_poins)
        cov_mat = np.zeros((n, n))
        for i in range(n):
            diff = raw_cov[i, i] + raw_cov[i + 1, i + 1] * (float(i+1)/float(i+2))

            diff -= 2.0 * raw_cov[i, i + 1] * np.sqrt((float(i+1)/float(i+2)))
            cov_mat[i, i] = diff
            for j in range(i + 1, n):
                diff = (raw_cov[i, j])  +\
                       (raw_cov[i + 1, j + 1] * np.sqrt((float(i+1)/float(i+2))) * np.sqrt((float(j+1)/float(j+2))))
                diff -= ((raw_cov[i, j + 1] * np.sqrt((float(j+1)/float(j+2))))+
                         (raw_cov[i + 1, j] * np.sqrt((float(i+1)/float(i+2)))))
                cov_mat[i, j] = diff
                cov_mat[j, i] = diff
        return cov_mat


    def log_likelihood(self, gp_model, parameters_kernel, mean_parameters=None, weights=None):
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
            mean = self.compute_parametric_mean(gp_model, weights, mean_parameters)
        else:
            mean = 0
        y_unbiased = gp_model.data['evaluations'] - mean
        solve = cho_solve(chol, y_unbiased)

        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(y_unbiased, solve)

    def compute_parametric_mean(self, gp_model, weights, mean_parameters):
        f = self.function_factor_kernel
        X_data = gp_model.data['points']
        mean_vector = np.zeros(len(X_data))
        for i in range(len(mean_vector)):
            val_1 = self.parametrics.weighted_combination(i + 1, weights, mean_parameters)
            val_2 = self.parametrics.weighted_combination(i + 2, weights, mean_parameters) * np.sqrt((i+1)/(i+2))
            mean_vector[i] = val_1 - val_2

        return mean_vector

    def log_prob(self, parameters, gp_model):
        """

        :param parameters: [kernel parameters, weights, mean parameters]
        :param gp_model:
        :return:
        """

        mean_parameters = None
        weights = None
        dim_kernel_params = gp_model.dimension_parameters
        kernel_parameters = parameters[0:dim_kernel_params]

        lp = 0.0
        parameters_model = gp_model.get_parameters_model[2:]
        index = 0

        for parameter in parameters_model:
            dimension = parameter.dimension
            lp += parameter.log_prior(kernel_parameters[index: index + dimension])
            index += dimension

        if not np.isinf(lp) and self.parametric_mean:
            weights = np.array(
                parameters[dim_kernel_params:dim_kernel_params+self.parametrics.n_weights])
            mean_parameters = parameters[self.parametrics.n_weights+dim_kernel_params:]
            lp += self.parametrics.log_prior(mean_parameters, weights)

        if not np.isinf(lp):
            lp += self.log_likelihood(gp_model, kernel_parameters, mean_parameters, weights)
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
                    # start_point_ = \
                    #     gp_model.slice_samplers[0].slice_sample(points, None, *(gp_model,))
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

        dimension_sampler = gp_model.dimension_parameters

        if self.parametric_mean:
            dimension_sampler += self.parametrics.n_weights + self.parametrics.n_parameters

        gp_model.slice_samplers.append(SliceSampling(
            self.log_prob, range(dimension_sampler), ignore_index=ignore_index,
            **slice_parameters))

        if gp_model.start_point_sampler is not None and len(gp_model.start_point_sampler) > 0:
            if len(gp_model.samples_parameters) == 0:
                gp_model.samples_parameters.append(np.array(gp_model.start_point_sampler))
        else:
            gp_model.samples_parameters = []
            z = gp_model.get_value_parameters_model
            z = z[2:]

            if self.parametric_mean:
                if gp_model.mean_params is None or len(gp_model.mean_params) == 0:
                    gp_model.mean_params = self.fit_parametric_functions(gp_model.data['evaluations'])
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
            cov[i] = raw_cov[0, i]
            cov[i] -= raw_cov[0, i+1] * np.sqrt(float(i+1)/float(i+2))
        return cov


    def compute_posterior_params(self, gp_model, kernel_params, mean_parameters=None, weights=None, iteration=None):
        ## Mean is zero. We may need to change when including mean
        ##Compute posterior parameters of f(x*)

        f = self.function_factor_kernel
        g = self.divisor_kernel

        if iteration is None:
            current_point = [gp_model.current_iteration]
        else:
            current_point = [iteration]

        X_data = gp_model.data['points']
        vector_ = self.cov_diff_point(gp_model, kernel_params, current_point, X_data)

        cov = self.covariance_diff_kernel(gp_model, X_data, kernel_params)
        chol = cholesky(cov, max_tries=7)

        if self.parametric_mean:
            mean = self.compute_parametric_mean(gp_model, weights, mean_parameters)
            prior_mean = self.parametrics.weighted_combination(
                current_point[0], weights, mean_parameters)
        else:
            mean = 0.
            prior_mean = 0.

        y_unbiased = gp_model.data['evaluations'] - mean
        solve = cho_solve(chol, y_unbiased)

        part_2 = cho_solve(chol, vector_)

        mean = np.dot(vector_, solve) + prior_mean

        # mean = gp_model.raw_results['values'][-1][0] - \
        #        np.array(gp_model.raw_results['gradients'][-1]) * (np.dot(vector_, solve) + prior_mean) \
        #        / np.sqrt(gp_model.current_iteration)

        raw_cov = gp_model.kernel.evaluate_cov_defined_by_params(
            kernel_params, np.array([current_point]), 1)

        var = raw_cov - np.dot(vector_, part_2)
        # var *= (np.array(gp_model.raw_results['gradients'][-1]) ** 2) / (float(gp_model.current_iteration))

        return mean, var[0, 0]

    def estimate_lipschitz(self, gp_model, mean, cov, kernel_params, mean_parameters=None, weights=None):
        if self.type_model == 'lipschitz':
            return self.lipschitz
        mean_, cov_ = self.folded_normal(mean, cov)

        diff = gp_model.raw_results['values'][-1] - gp_model.raw_results['values'][-2]

        mean_2, cov_2 = self.compute_posterior_params(
            gp_model, kernel_params, mean_parameters, weights, iteration=gp_model.current_iteration-1)
        mean__, cov__ = self.folded_normal(mean_2, cov_2)
        quotient = (1.0/np.sqrt(gp_model.current_iteration)) * mean_ - (1.0 / np.sqrt(gp_model.current_iteration - 1)) * mean__

        return diff / quotient

    def folded_normal(self, mean, cov):
        std = np.sqrt(cov)
        mean_ = std * np.sqrt(2.0 / np.pi) * np.exp(- (mean ** 2) / (2.0 * cov))
        mean_ += mean * (1.0 - 2.0 * norm.cdf(- mean / std))

        var = (mean ** 2) + cov - (mean_ ** 2)
        return mean_, var

    def compute_posterior_params_marginalize(self, gp_model, n_samples=10, burning_parameters=True, get_vectors=False):
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
            mean_parameters = None
            weights = None

            if self.parametric_mean:
                mean_params = param[dim_kernel_params:]
                weights = mean_params[0:self.parametrics.n_weights]
                mean_parameters = mean_params[self.parametrics.n_weights:]

            mean, cov = self.compute_posterior_params(
                gp_model, kernel_params, mean_parameters, weights)
            mean_, cov_ = self.folded_normal(mean, cov)
            L = self.estimate_lipschitz(gp_model, mean, cov, kernel_params, mean_parameters, weights)

            # TODO: NOT SURE IF USING NP.ABS(L) or not
            mean = gp_model.raw_results['values'][-1][0] + (np.abs(L) * mean_) / np.sqrt(gp_model.current_iteration)
            cov = cov_ * (L ** 2) / (gp_model.current_iteration)
            means.append(mean)
            covs.append(cov)

        mean = np.mean(means)
        std = np.sqrt(np.mean(covs))

        ci = [mean - 1.96 * std, mean + 1.96 * std]
        if get_vectors:
            means = [t - gp_model.raw_results['values'][-1][0] for t in means]
            return {'means': means, 'covs': covs, 'value': gp_model.raw_results['values'][-1][0]}

        return mean, std, ci

    def add_observations(self, gp_model, point, y, new_point_in_domain=None, gradient=None,
                         model=None):
        gp_model.current_iteration = point
        gp_model.best_result = max(gp_model.best_result, y)
        gp_model.data['points'].append((point - 1, point))

        previous_x = np.array(gp_model.raw_results['points'][-1])

        if new_point_in_domain is not None:
            gp_model.raw_results['points'].append(new_point_in_domain)
        gp_model.raw_results['values'].append(y)
        if gradient is not None:
            gp_model.raw_results['gradients'].append(gradient)


        gp_model.data['evaluations'] = np.concatenate(
            (gp_model.data['evaluations'], np.sqrt(point - 1) * (previous_x - new_point_in_domain)))


        if self.current_batch_index + 1 > self.total_batches:
            self.current_epoch += 1

        self.current_batch_index = (self.current_batch_index + 1) % self.total_batches

        if new_point_in_domain is not None:
            self.current_point = new_point_in_domain


    def accuracy(self, gp_model, start=3, iterations=21, sufix=None, model=None):
        means = {}
        cis = {}

        mean, std, ci = self.compute_posterior_params_marginalize(gp_model)
        means[start] = mean
        cis[start] = ci

        for i in range(start, iterations):
            print (i)
            if len(gp_model.raw_results) < i + 1:
                data_new = self.get_value_next_iteration(i+1, **self.kwargs)
                self.add_observations(gp_model, i+1, data_new['value'], data_new['point'], data_new['gradient'])
            mean, std, ci = self.compute_posterior_params_marginalize(gp_model)
            means[i + 1] = mean
            cis[i + 1] = ci

            print mean, ci
            value_tmp = self.get_value_next_iteration(i+1, **self.kwargs)
            print value_tmp

        accuracy_results = {}
        accuracy_results['means'] = means
        accuracy_results['ci'] = cis
        file_name = 'data/multi_start/accuracy_results/stat_model'

        if not os.path.exists('data/multi_start'):
            os.mkdir('data/multi_start')

        if not os.path.exists('data/multi_start/accuracy_results'):
            os.mkdir('data/multi_start/accuracy_results')

        if self.problem_name is not None:
            file_name += '_' + self.problem_name

        if sufix is not None:
            file_name += '_' + sufix

        JSONFile.write(accuracy_results, file_name + '.json')

        return means, cis

    def plot_accuracy_results(self, means, cis, original_value, start=3, final_iteration=10, sufix=None, n_epoch=1):
        plt.figure()
        x_lim = len(means)


        means_vec = []
        cis_vec = []
        points = []

        for i in sorted(means):
            points.append(i)
            means_vec.append(means[i])
            cis_vec.append(cis[i])

       # points = range(start, final_iteration, n_epoch)
        plt.plot(points, means_vec, 'b', label='means')
        plt.plot(points, len(points) * [original_value], label='final value')
        plt.plot(points, [t[0] for t in cis_vec],'g-', label='ci')
        plt.plot(points, [t[1] for t in cis_vec],'g-', label='ci')

        plt.legend()
        plt.ylabel('Objective function')
        plt.xlabel('Iteration')

        file_name = 'data/multi_start/accuracy_plots/stat_model'

        if not os.path.exists('data/multi_start'):
            os.mkdir('data/multi_start')

        if not os.path.exists('data/multi_start/accuracy_plots'):
            os.mkdir('data/multi_start/accuracy_plots')

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

    def log_prob_per_parametric_function(self, x, historical_data, function_name, weight=1.0):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x: dictionary with arguments of function
        historical_data: [float]
        """

        f = self.function_factor_kernel

        function = self.parametrics.functions[function_name]
        params = x

        dom_x = range(1, len(historical_data) + 1)
        evaluations = np.zeros(len(historical_data))
        for i in dom_x:
            first_1 = weight * function(i, **params)
            first_2 = weight * function(i + 1, **params) * np.sqrt((i) / (i+1))
            # val = -np.log(first_1) / f(i)
            # val -= -np.log(first_2) / f(i + 1)
            evaluations[i - 1] = first_1 - first_2

        val = -1.0 * np.sum((evaluations - historical_data) ** 2)

        return val

    def log_prob_all_parametric_function(self, x, historical_data):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x: [weights, parameters]
        historical_data: [float]
        """

        f = self.function_factor_kernel

        weights = x[0: self.parametrics.n_weights]
        parameters = x[self.parametrics.n_weights:]

        dom_x = range(1, len(historical_data) + 1)
        evaluations = np.zeros(len(historical_data))
        for i in dom_x:
            first_1 = self.parametrics.weighted_combination(i, weights, parameters)
            first_2 = self.parametrics.weighted_combination(i+1, weights, parameters) * np.sqrt((i) / (i+1))
            # val = -np.log(first_1) / f(i)
            # val -= -np.log(first_2) / f(i + 1)
            evaluations[i - 1] = first_1 - first_2

        val = -1.0 * np.sum((evaluations - historical_data) ** 2)

        return val

    def gradient_llh_all_functions(self, x, historical_data):
        f = self.function_factor_kernel

        weights = x[0: self.parametrics.n_weights]
        parameters = x[self.parametrics.n_weights:]

        evaluations = np.zeros(len(historical_data))

        dom_x = range(1, len(historical_data) + 1)

        gradient = np.zeros(self.parametrics.n_parameters + self.parametrics.n_weights)

        for i in dom_x:
            # evaluations[i - 1] = -np.log(weight * function(i, **params)) / f(i)
            # evaluations[i - 1] -= -np.log(weight * function(i + 1, **params)) / f(i + 1)
            # tmp = - weight * gradient_function(i, **params) / (f(i) * weight * function(i, **params))
            # tmp -= -weight * gradient_function(i + 1, **params) / (
            # f(i + 1) * weight * function(i + 1, **params))
            # gradient_theta += tmp * (evaluations[i - 1] - historical_data[i - 1])
            tmp = self.parametrics.gradient_weighted_combination(i, weights, parameters)
            tmp -= self.parametrics.gradient_weighted_combination(i+1, weights, parameters) * np.sqrt((i) / (i+1))

            first_1 = self.parametrics.weighted_combination(i, weights, parameters)
            first_2 = self.parametrics.weighted_combination(i+1, weights, parameters) * np.sqrt((i) / (i+1))

            evaluations[i-1] = first_1 - first_2

            gradient += tmp * (evaluations[i-1] - historical_data[i-1])

        gradient *= -2.0

        return gradient

    def mle_params_all_functions(
            self, historical_data, lower=0.0, upper=1.0, total_iterations=100):
        """
        log_prob for only one starting point

        :param historical_data: [float]
        """
        historical_data = np.array(historical_data)
        n = len(historical_data)

        def objective(params):
            val = self.log_prob_all_parametric_function(params, historical_data)
            return -1.0 * val

        def gradient(params):
            val = self.gradient_llh_all_functions(params, historical_data)
            return -1.0 * val

        params_st = self.parametrics.get_starting_values()
        bounds = self.parametrics.get_bounds()

        popt, fval, info = fmin_l_bfgs_b(
            objective, fprime=gradient, x0=params_st, bounds=bounds, approx_grad=False)

        weights = popt[0: self.parametrics.n_weights]
        params = popt[self.parametrics.n_weights:]

        if lower is not None:
            value_1 = self.parametrics.weighted_combination(1, weights, params)
            if value_1 < lower:
                w = lower / value_1
                popt[0: self.parametrics.n_weights] = w

        if upper is not None:
            value_2 = self.parametrics.weighted_combination(total_iterations, weights, params)
            if value_2 > upper:
                w = upper / value_2
                popt[0: self.parametrics.n_weights] = w

        return popt, fval, info

    def fit_parametric_functions(self, historical_data):
        lower = self.parametrics.lower
        upper = self.parametrics.upper
        total_iterations = self.parametrics.total_iterations

        params = self.mle_params_all_functions(historical_data, lower, upper, total_iterations)

        return params[0]

    def gradient_llh_per_function(self, x, historical_data, function_name, weight=1.0):
        """
        Gradient of the llh respect to a specific function.

        :param function: str
        """
        f = self.function_factor_kernel

        function = self.functions[function_name]
        gradient_function = self.gradients_functions[function_name]
        params = x

        evaluations = np.zeros(len(historical_data))

        dom_x = range(1, len(historical_data) + 1)
        gradient_theta = np.zeros(len(x))
        gradient = np.zeros(len(x) + 1)
        gradient_weight = 0.0
        for i in dom_x:
            # evaluations[i - 1] = -np.log(weight * function(i, **params)) / f(i)
            # evaluations[i - 1] -= -np.log(weight * function(i + 1, **params)) / f(i + 1)
            # tmp = - weight * gradient_function(i, **params) / (f(i) * weight * function(i, **params))
            # tmp -= -weight * gradient_function(i + 1, **params) / (
            # f(i + 1) * weight * function(i + 1, **params))
            # gradient_theta += tmp * (evaluations[i - 1] - historical_data[i - 1])

            tmp = weight * gradient_function(i, **params) / f(i)
            tmp -= weight * gradient_function(i + 1, **params) / f(i+1)

            evaluations[i-1] = weight * function(i, **params) / f(i)
            evaluations[i-1] -= weight * function(i + 1, **params) / f(i+1)

            gradient_theta += tmp * (evaluations[i-1] - historical_data[i-1])

            tmp_grad = function(i, **params) / f(i)
            tmp_grad -= function(i+1, **params) / f(i+1)

            gradient_weight += tmp_grad * (evaluations[i-1] - historical_data[i-1])

        gradient_theta *= -2.0

        gradient[0:-1] = gradient_theta
        gradient[-1] = -2.0 * gradient_weight

        return gradient

