from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
import sys
import os
import matplotlib.pyplot as plt

from multi_start.parametric_functions import ParametricFunctions
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling


class ParametricModel(object):

    def __init__(self, raw_results, best_result, current_iteration, get_value_next_iteration,
                 starting_point, current_batch_index, current_epoch, problem_name=None,
                 max_iterations=1000,
                 lower=None, upper=None, n_burning=500, thinning=10, total_batches=10,
                 kwargs_get_value_next_iteration=None):

        self.raw_results = raw_results
        self.best_result = best_result
        self.current_iteration = current_iteration
        self.get_value_next_iteration = get_value_next_iteration
        self.starting_point = starting_point
        self.current_batch_index = current_batch_index
        self.problem_name = problem_name
        self.max_iterations = max_iterations
        self.lower = lower
        self.upper = upper
        self.total_batches = total_batches

        self.current_point = starting_point
        self.current_epoch = current_epoch

        self.n_burning = n_burning
        self.thinning = thinning

        self.training_data = None
        self.process_data()

        self.parametrics = ParametricFunctions(max_iterations, lower, upper)

        self.slice_samplers = []
        self.samples_parameters = []
        self.start_point_sampler = []
        self.set_samplers()

        self.kwargs = {}
        if kwargs_get_value_next_iteration is not None:
            self.kwargs = kwargs_get_value_next_iteration

    def process_data(self):
        training_data = {}
        training_data['points'] = [[float(i)] for i in range(1, len(self.raw_results) + 1)]
        training_data['evaluations'] = list(self.raw_results)
        training_data['var_noise'] = []
        self.training_data = training_data


    def log_likelihood(self, parameters, weights, sigma):
        val = 0.0
        historical_data = self.training_data['evaluations']
        for index, y in enumerate(historical_data):
            mean = self.parametrics.weighted_combination(index + 1, weights, parameters)
            val += np.log(norm.pdf(y, loc=mean, scale=sigma))
        return val

    def log_prob(self, parameters):
        """

        :param parameters: [weights, mean parameters, sigma]
        :return:
        """

        lp = 0.0

        weights = np.array(parameters[0:self.parametrics.n_weights])
        mean_parameters = parameters[self.parametrics.n_weights:-1]
        sigma = parameters[-1]
        lp += self.parametrics.log_prior(mean_parameters, weights)

        if not np.isinf(lp):
            lp += self.log_likelihood(mean_parameters, weights, sigma)

        return lp

    def log_prob_all_parametric_function(self, x, historical_data):
        """
        log_prob for only one starting point
        Historical data of only one starting point.

        x: [weights, parameters]
        historical_data: [float]
        """
        weights = x[0: self.parametrics.n_weights]
        parameters = x[self.parametrics.n_weights:]

        dom_x = range(1, len(historical_data) + 1)
        evaluations = np.zeros(len(historical_data))

        for i in dom_x:
            value = self.parametrics.weighted_combination(i, weights, parameters)
            evaluations[i - 1] = value

        val = -1.0 * np.sum((evaluations - historical_data) ** 2)

        return val

    def gradient_llh_all_functions(self, x, historical_data):
        weights = x[0: self.parametrics.n_weights]
        parameters = x[self.parametrics.n_weights:]

        evaluations = np.zeros(len(historical_data))

        dom_x = range(1, len(historical_data) + 1)

        gradient = np.zeros(self.parametrics.n_parameters + self.parametrics.n_weights)

        for i in dom_x:
            tmp = self.parametrics.gradient_weighted_combination(i, weights, parameters)

            first_1 = self.parametrics.weighted_combination(i, weights, parameters)

            evaluations[i-1] = first_1

            gradient += tmp * (evaluations[i-1] - historical_data[i-1])

        gradient *= -2.0

        return gradient

    def mle_params_all_functions(self, training_data, lower, upper, total_iterations):

        historical_data = np.array(training_data)

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

        domain_x = range(1, len(training_data))
        evaluations = np.array(
            [self.parametrics.weighted_combination(t, weights, params) for t in domain_x])
        sigma_sq = np.mean((evaluations - training_data) ** 2)
        popt = np.concatenate((popt, np.sqrt(sigma_sq)))
        return popt, fval, info

    def fit_parametric_functions(self, training_data):
        lower = self.parametrics.lower
        upper = self.parametrics.upper
        total_iterations = self.parametrics.total_iterations

        params = self.mle_params_all_functions(training_data, lower, upper, total_iterations)

        return params[0]

    def set_samplers(self):

        self.slice_samplers = []
        self.samples_parameters = []
        self.start_point_sampler = []

        slice_parameters = {
            'max_steps_out': 1000,
            'component_wise': False,
        }

        n_params = self.parametrics.n_parameters + self.parametrics.n_weights + 1
        sampler = SliceSampling(
            self.log_prob, range(0, len(n_params)), **slice_parameters)
        self.slice_samplers.append(sampler)

        if self.start_point_sampler is not None and len(self.start_point_sampler) > 0:
            if len(self.samples_parameters) == 0:
                self.samples_parameters.append(np.array(self.start_point_sampler))
        else:
            self.samples_parameters = []
            start_point = self.fit_parametric_functions(self.training_data['evaluations'])
            self.samples_parameters.append(start_point)

            if self.n_burning > 0:
                parameters = self.sample_parameters(
                    self.slice_samplers[0], float(self.n_burning) / (self.thinning + 1))
                self.samples_parameters = []
                self.samples_parameters.append(parameters[-1])
                self.start_point_sampler = parameters[-1]
            else:
                self.start_point_sampler = self.samples_parameters[-1]

    def sample_parameters(self,  sampler, n_samples, start_point=None, random_seed=None):
        """
        Sample parameters of the model from the posterior without considering burning.

        :param n_samples: (int)
        :param start_point: np.array(n_parameters)
        :param random_seed: intf

        :return: n_samples * [np.array(float)]
        """

        if random_seed is not None:
            np.random.seed(random_seed)


        samples = []

        if start_point is None:
            start_point = self.samples_parameters[-1]

        n_samples *= (self.thinning + 1)
        n_samples = int(n_samples)

        for sample in range(n_samples):
            start_point_ = None
            n_try = 0
            while start_point_ is None and n_try < 10:
                #  start_point_ = sampler.slice_sample(start_point, None)
                try:
                    start_point_ = sampler.slice_sample(start_point, None)
                except Exception as e:
                    n_try += 1
                    start_point_ = None
            if start_point_ is None:
                print('program failed to compute a sample of the parameters')
                sys.exit(1)
            start_point = start_point_
            samples.append(start_point)
        return samples[::self.thinning + 1]

    def add_observations(self, point, y, new_point_in_domain=None):
        self.current_iteration = point
        self.raw_results.append(y)
        self.best_result = max(self.best_result, y)
        self.training_data['points'].append(point)
        self.training_data['evaluations'] = np.concatenate(
            (self.training_data['evaluations'], [y]))

        if self.current_batch_index + 1 > self.total_batches:
            self.current_epoch += 1

        self.current_batch_index = (self.current_batch_index + 1) % self.total_batches

        if new_point_in_domain is not None:
            self.current_point = new_point_in_domain

    def compute_posterior_params_marginalize(self, n_samples=10, burning_parameters=True):
        if burning_parameters:
            parameters = self.sample_parameters(
                self.slice_samplers[0], float(self.n_burning) / (self.thinning + 1))
            self.samples_parameters = []
            self.samples_parameters.append(parameters[-1])
            self.start_point_sampler = parameters[-1]

        parameters = self.sample_parameters(self.slice_samplers[0], n_samples)

        means = []
        covs = []
        for param in parameters:
            weights = param[0: self.parametrics.n_weights]
            params = param[self.parametrics.n_weights:-1]
            var = param[-1] ** 2
            parametric_mean = self.parametrics.weighted_combination(
                self.max_iterations, weights, params)
            means.append(parametric_mean)
            covs.append(var)

        mean = np.mean(means)
        std = np.sqrt(np.mean(covs))

        ci = [mean - 1.96 * std, mean + 1.96 * std]

        return mean, std, ci

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
        file_name = 'data/multi_start/accuracy_results/parametric_model'

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

        file_name = 'data/multi_start/accuracy_plots/parametric_model'

        if not os.path.exists('data/multi_start'):
            os.mkdir('data/multi_start')

        if not os.path.exists('data/multi_start/accuracy_plots'):
            os.mkdir('data/multi_start/accuracy_plots')

        if self.problem_name is not None:
            file_name += '_' + self.problem_name

        if sufix is not None:
            file_name += '_' + sufix
        plt.savefig(file_name + '.pdf')