from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit, leastsq, fmin_bfgs, fmin_l_bfgs_b, nnls
import sys
import os
import matplotlib.pyplot as plt

from stratified_bayesian_optimization.lib.la_functions import (
    cholesky,
    cho_solve,
)

from multi_start.parametric_functions import ParametricFunctions
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling


class StatModelSwersky(object):

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
        self.n_thinning = thinning
        self.thinning = thinning
        self.burning = True

        self.training_data = None
        self.process_data()

        self.gp_model = None
        self.define_gp_model()

        self.slice_samplers = []
        self.samples_parameters = []
        self.start_point_sampler = []
        self.set_samplers()

        self.kwargs = {}
        if kwargs_get_value_next_iteration is not None:
            self.kwargs = kwargs_get_value_next_iteration

    def process_data(self):
        training_data = {}
        training_data['points'] = [[float(i)] for i in range(0, len(self.raw_results['values']))]
        training_data['evaluations'] = list(self.raw_results['values'])
        training_data['var_noise'] = []
        self.training_data = training_data

    def define_gp_model(self):

        def toy_objective_function(x):
            return [np.sum(x)]

        spec = {
            'name_model': 'gp_fitting_gaussian',
            'problem_name': self.problem_name,
            'type_kernel': ['swersky_kernel'],
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
            'cache': False,
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
        model.data['points'] = [[float(i)] for i in range(0, len(self.raw_results['values']))]
        model.mean_params = []

        self.gp_model = model


    def log_likelihood(self, gp_model, kernel_parameters):

        X_data = gp_model.data['points']

        cov = gp_model.kernel.evaluate_cov_defined_by_params(kernel_parameters, X_data, 1)
        chol = cholesky(cov, max_tries=7)


        y_unbiased = gp_model.data['evaluations']


        solve = cho_solve(chol, y_unbiased)

        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(y_unbiased, solve)

    def log_prob(self, parameters):
        """

        :param parameters: [kernel parameters]
        :param gp_model:
        :return:
        """
        gp_model = self.gp_model
        dim_kernel_params = gp_model.dimension_parameters
        kernel_parameters = parameters[0:dim_kernel_params]

        lp = 0.0
        parameters_model = gp_model.get_parameters_model[2:]
        index = 0

        for parameter in parameters_model:
            dimension = parameter.dimension
            lp += parameter.log_prior(kernel_parameters[index: index + dimension])
            index += dimension

        if not np.isinf(lp):
            lp += self.log_likelihood(gp_model, kernel_parameters)
        return lp



    def set_samplers(self):
        gp_model = self.gp_model
        gp_model.slice_samplers = []
        gp_model.samples_parameters = []
        gp_model.start_point_sampler = []

        ignore_index = None

        slice_parameters = {
            'max_steps_out': gp_model.max_steps_out,
            'component_wise': True,
        }

        dimension_sampler = gp_model.dimension_parameters


        gp_model.slice_samplers.append(SliceSampling(
            self.log_prob, range(dimension_sampler), ignore_index=ignore_index,
            **slice_parameters))

        if gp_model.start_point_sampler is not None and len(gp_model.start_point_sampler) > 0:
            if len(gp_model.samples_parameters) == 0:
                gp_model.samples_parameters.append(np.array(gp_model.start_point_sampler))
        else:
            gp_model.samples_parameters = []
            self.samples_parameters = []
            z = gp_model.get_value_parameters_model
            z = z[2:]

            gp_model.samples_parameters.append(z)
            self.samples_parameters.append(z)

            if gp_model.n_burning > 0 and self.burning:
                parameters = self.sample_parameters(
                    gp_model, float(gp_model.n_burning) / (gp_model.thinning + 1))
                self.samples_parameters = []
                self.samples_parameters.append(parameters[-1])
                self.start_point_sampler = parameters[-1]

                gp_model.samples_parameters = []
                gp_model.samples_parameters.append(parameters[-1])
                gp_model.start_point_sampler = parameters[-1]
            else:
                gp_model.start_point_sampler = gp_model.samples_parameters[-1]
                self.start_point_sampler = gp_model.samples_parameters[-1]

    def sample_parameters(self,  gp_model, n_samples, start_point=None, random_seed=None):
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
                start_point_ = gp_model.slice_samplers[0].slice_sample(start_point, None)
                try:
                    start_point_ = gp_model.slice_samplers[0].slice_sample(start_point, None)
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
        y = y[0]

        gp_model = self.gp_model


        gp_model.current_iteration = point
        gp_model.best_result = max(gp_model.best_result, y)
        gp_model.data['points'].append([float(point-1)])

        gp_model.data['evaluations'] = np.concatenate(
            (gp_model.data['evaluations'], [y]))

        self.current_iteration = point
    #    self.raw_results['values'].append(y)

        self.best_result = max(self.best_result, y)
        # self.training_data['points'].append([float(point+1)])
        # self.training_data['evaluations'] = np.concatenate(
        #     (self.training_data['evaluations'], [y]))

        if self.current_batch_index + 1 > self.total_batches:
            self.current_epoch += 1

        self.current_batch_index = (self.current_batch_index + 1) % self.total_batches

        if new_point_in_domain is not None:
            self.current_point = new_point_in_domain





    def compute_posterior_params_marginalize(self, n_samples=10, burning_parameters=True):
        gp_model = self.gp_model
        if burning_parameters:

            parameters = self.sample_parameters(
                gp_model, float(self.n_burning) / (self.n_thinning + 1))
            self.samples_parameters = []
            self.samples_parameters.append(parameters[-1])
            self.start_point_sampler = parameters[-1]

            gp_model.samples_parameters = []
            gp_model.samples_parameters.append(parameters[-1])
            gp_model.start_point_sampler = parameters[-1]

        parameters = self.sample_parameters(gp_model, n_samples)

        means = []
        covs = []
        dim_kernel_params = gp_model.dimension_parameters
        for param in parameters:
            kernel_params = param[0: dim_kernel_params]

            result = gp_model.compute_posterior_parameters(np.array([[float(self.max_iterations)]]), var_noise=0.0, mean=0.0,
                                         parameters_kernel=kernel_params, only_mean=False)
            mean =result['mean']
            cov= result['cov']

            means.append(mean)
            covs.append(cov)

        return {'means': means, 'covs': covs, 'value': 0.0}

        # mean = np.mean(means)
        # std = np.sqrt(np.mean(covs))
        #
        # ci = [mean - 1.96 * std, mean + 1.96 * std]
        #
        # return mean, std, ci

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
        file_name = 'data/multi_start/accuracy_results/swk'

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