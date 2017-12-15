from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from scipy.stats import norm
from random import shuffle
from scipy.optimize import brute
from copy import deepcopy

import itertools

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.util import (
    wrapper_objective_acquisition_function,
)
from stratified_bayesian_optimization.lib.la_functions import (
    cholesky,
    cho_solve,
)

logger = SBOLog(__name__)


class SDE(object):


    def __init__(self, gp, n_tasks, domain_xe, x_domain, weights):
        """
        See http://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n46.pdf
        Only uses Matern Kernel
        :param gp:
        :param n_tasks:
        """
        self.gp = gp
        self.n_tasks = n_tasks
        self.domain_xe = domain_xe # range of values as a list
        self.x_domain = x_domain # [0, 1,2, .., x_domain-1] dimension of domain of x
        self.weights = weights

    def estimate_variance_gp(self, parameters_kernel, chol=None):
        historical_points = self.data['points']
        if chol is None:
            cov = self.gp.evaluate_cov(historical_points, parameters_kernel)
            chol = cholesky(cov, max_tries=7)

        y = self.gp.data['evaluations']
        n = chol.shape[0]
        z = np.ones(n)

        solve = cho_solve(chol, y)
        part_1 = np.dot(y, solve)

        solve_2 = cho_solve(chol, z)
        part_2 = np.dot(z, solve_2)

        beta = np.dot(z, solve) / part_2
        part_2 *= (beta ** 2)

        return (part_1 - part_2 ) / (n - 1), beta


    def log_posterior_distribution_length_scale(self, parameters_kernel):
        historical_points = self.data['points']
        cov = self.gp.evaluate_cov(historical_points, parameters_kernel)

        n = cov.shape[0]
        y = np.ones(n)
        chol = cholesky(cov, max_tries=7)
        determinant_cov = np.product(np.diag(chol)) ** 2

        solve = cho_solve(chol, y)
        part_1 = np.dot(y, solve)
        var = self.estimate_variance_gp(parameters_kernel, chol=chol)[0]

        objective = \
            -(n-1) * 0.5 * np.log(var) - 0.5 * np.log(determinant_cov) - 0.5 * np.log(part_1)

        return -1.0 * objective


    def sample_variable(self, parameters_kernel, n_samples):
        historical_points = self.gp.data['points']
        var, beta = self.estimate_variance_gp(parameters_kernel)
        y = self.gp.data['evaluations']
        n = len(y)
        z = np.ones(n)

        create_vector = np.zeros((historical_points.shape[0] * len(self.domain_xe), y.shape[1]))

        for i in xrange(historical_points.shape[0]):
            for j in xrange(len(self.domain_xe)):
                first_part = historical_points[i][0:self.x_domain]
                point = np.concatenate((first_part, np.array(self.domain_xe[j])))
                create_vector[(i-1)*len(self.domain_xe) + j] = point

        cov = self.gp.evaluate_cov(historical_points, parameters_kernel)
        chol = cholesky(cov, max_tries=7)

        matrix_cov = self.gp.kernel.evaluate.evaluate_cross_cov_defined_by_params(
            parameters_kernel, self.gp.data, create_vector)
        solve = np.dot(matrix_cov.transpose() ,cho_solve(chol, y - beta * z))

        mean = z * beta + solve


        chi = np.random.chisq(n-1, n_samples)
        chi = (n-1) * var / chi

        samples = []
        for i in xrange(n_samples):
            sample = np.random.multivariate_normal(mean, chi[i] * cov)
            samples.append(sample)

        return samples

    def ei_given_sample(self, sample, parameters_kernel):
        M = np.min(sample)
        n = len(sample)
        one = np.ones(2 * n)
        historical_points = self.gp.data['points']
        y = self.gp.data['evaluations']
        Z = np.concatenate((y, sample))


        create_vector = np.zeros((historical_points.shape[0] * len(self.domain_xe), y.shape[1]))

        for i in xrange(historical_points.shape[0]):
            for j in xrange(len(self.domain_xe)):
                first_part = historical_points[i][0:self.x_domain]
                point = np.concatenate((first_part, np.array(self.domain_xe[j])))
                create_vector[(i-1)*len(self.domain_xe) + j] = point

        new_vector = np.concatenate((historical_points, create_vector))

        C = self.gp.kernel.evaluate.evaluate_cross_cov_defined_by_params(
            parameters_kernel, new_vector, new_vector)
        chol = cholesky(C, max_tries=7)

        solv = cho_solve(chol, Z)
        solv_2 = cho_solve(chol, one)

        bc = np.dot(one, solv) / np.dot(one, solv_2)

        c = np.dot(self.weights, )

        part_1 = np.dot(c, cho_solve(chol, Z - bc * one))
        mc = bc  +



    def ei_objective(self, x, samples):



    def control_variable_ei(self, parameters_kernel):
        samples = self.sample_variable(parameters_kernel, 100)



    def optimize(self, **kwargs):
        parameters = brute(self.log_posterior_distribution_length_scale, ranges=())



    def estimate_length_scale(self):



    def add_file_to_log(self, model_type, problem_name, training_name, n_training, random_seed,
                        n_samples_parameters):
        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        logger.add_file_to_log(model_type, problem_name, kernel_name, training_name, n_training,
                               random_seed, n_samples_parameters)

    def evaluate_first(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Computes the ei after imputing missing observations using the predictive means.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: float
        """
        return self.ei.evaluate(point, var_noise, mean, parameters_kernel)

    def optimize_first(self, start=None, random_seed=None, parallel=True, n_restarts=100,
                       n_samples_parameters=0, n_best_restarts=0, maxepoch=11):
        """
        Optimizes EI

        :param start: np.array(n)
        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: int
        :param n_samples_parameters int
        :param n_best_restarts: int

        :return np.array(n)
        """

        solution = self.ei.optimize(start, random_seed, parallel, n_restarts,
                                    n_best_restarts=n_best_restarts,
                                    n_samples_parameters=n_samples_parameters, maxepoch=maxepoch)

        return solution['solution']

    def choose_best_task_given_x(self, x, n_samples_parameters=0):
        """

        :param x: np.array(n)
        :param n_samples_parameters: int
        :return: int
        """
        values = []
        for i in xrange(self.n_tasks):
            point = np.concatenate((x, np.array([i])))
            val = wrapper_objective_acquisition_function(point, self.ei_tasks, n_samples_parameters)
            values.append(val)

        return np.argmax(values), values[np.argmax(values)]

    def optimize(self, random_seed=None, parallel=True, n_restarts=100, n_best_restarts=0,
                 n_samples_parameters=0, start_new_chain=True, maxepoch=11, **kwargs):
        """
        Optimizes EI

        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: int
        :param n_best_restarts: int
        :param n_samples_parameters: int
        :param start_new_chain: boolean

        :return {'solution': np.array(n)}
        """

        if n_samples_parameters > 0 and start_new_chain:
            self.bq.gp.start_new_chain()
            self.bq.gp.sample_parameters(n_samples_parameters)

        point = self.optimize_first(random_seed=random_seed, parallel=parallel,
                                    n_restarts=n_restarts,
                                    n_samples_parameters=n_samples_parameters,
                                    n_best_restarts=n_best_restarts, maxepoch=maxepoch)

        task, value = self.choose_best_task_given_x(point, n_samples_parameters=n_samples_parameters)

        solution = np.concatenate((point, np.array([task])))

        return {'solution': solution, 'optimal_value': value}

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         n_samples_parameters, **kwargs):
        self.ei.write_debug_data(problem_name, model_type, training_name, n_training, random_seed,
                                 n_samples_parameters)

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.ei_tasks.clean_cache()
        self.ei.clean_cache()
