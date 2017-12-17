from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from scipy.stats import norm
from random import shuffle
from scipy.optimize import brute
from copy import deepcopy
import scipy.optimize

import itertools

from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.constant import (
    LBFGS_NAME, SGD_NAME, NEWTON_CG_NAME, TRUST_N_CG, DOGLEG, NELDER, SMALLEST_POSITIVE_NUMBER)
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.util import (
    wrapper_objective_acquisition_function,
    wrapper_optimize,
    wrapper_mean_objective,
    wrapper_ei_objective,
    wrapper_evaluate_squared_error,
)
from stratified_bayesian_optimization.lib.util import \
    wrapper_log_posterior_distribution_length_scale
from stratified_bayesian_optimization.lib.la_functions import (
    cholesky,
    cho_solve,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from scipy.stats import t

logger = SBOLog(__name__)


class SDE(object):


    def __init__(self, gp, domain_xe, x_domain, weights):
        """
        See http://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n46.pdf
        Only uses Matern Kernel
        :param gp:
        :param n_tasks:
        """
        self.gp = gp
        self.domain_xe = domain_xe # range of values as a list
        self.x_domain = x_domain # [0, 1,2, .., x_domain-1] dimension of domain of x
        self.w_domain = [i for i in range(self.gp.dimension_domain) if i not in range(x_domain)]
        self.weights = weights
        self.parameters = None

    def estimate_variance_gp(self, parameters_kernel, chol=None):
        """
        Correct
        :param parameters_kernel:
        :param chol:
        :return:
        """
        historical_points = self.gp.data['points']
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

        return (part_1 - part_2) / float(n - 1), beta


    def log_posterior_distribution_length_scale(self, parameters_kernel):
        """
        Correct
        :param parameters_kernel:
        :return:
        """
        historical_points = self.gp.data['points']
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

        return objective

    def compute_expectation_sample(self, parameters_kernel):
        """
        Correct
        :param parameters_kernel:
        :return:
        """
        historical_points = self.gp.data['points']
        var, beta = self.estimate_variance_gp(parameters_kernel)
        y = self.gp.data['evaluations']
        n = len(y)
        z = np.ones(n)

        create_vector = np.zeros(
            (historical_points.shape[0] * len(self.domain_xe), historical_points.shape[1]))

        for i in range(historical_points.shape[0]):
            for j in range(len(self.domain_xe)):
                first_part = historical_points[i][0:self.x_domain]
                point = np.concatenate((first_part, np.array(self.domain_xe[j])))
                create_vector[(i-1)*len(self.domain_xe) + j, :] = point

        cov = self.gp.evaluate_cov(historical_points, parameters_kernel)
        chol = cholesky(cov, max_tries=7)

        matrix_cov = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, self.gp.data['points'], create_vector, historical_points.shape[1])
        matrix_cov = np.dot(
            matrix_cov, np.kron(np.identity(n), self.weights.reshape((len(self.weights),1))))
        vect = y - beta * z
        vect = vect.reshape((len(vect), 1))
        solve = np.dot(matrix_cov.transpose() ,cho_solve(chol, vect))

        sample = np.ones((n, 1)) * beta + solve
        return sample


    def sample_variable(self, parameters_kernel, n_samples):
        """
        Correct
        See p.1141, we do (9)
        :param parameters_kernel:
        :param n_samples:
        :return:
        """
        y = self.gp.data['evaluations']
        historical_points = self.gp.data['points']
        n = len(y)
        var, beta = self.estimate_variance_gp(parameters_kernel)
        mean = self.compute_expectation_sample(parameters_kernel)
        mean = mean[:, 0]
        cov = self.gp.evaluate_cov(historical_points, parameters_kernel)

        chi = np.random.chisquare(n-1, n_samples)
        chi = (n-1) * var / chi

        samples = []
        for i in xrange(n_samples):
            sample = np.random.multivariate_normal(mean, chi[i] * cov)
            samples.append(sample)

        return samples

    def compute_mc_given_sample(self, sample, candidate_point, parameters_kernel):
        """
        Correct
        :param sample:
        :param candidate_point:
        :param parameters_kernel:
        :return:
        """
        if len(sample.shape) == 2:
            sample = sample[:, 0]

        n = len(sample)
        one = np.ones(2 * n)
        historical_points = self.gp.data['points']
        y = self.gp.data['evaluations']

        Z = np.concatenate((y, sample))

        cov_1 = self.gp.evaluate_cov(historical_points, parameters_kernel)


        create_vector = np.zeros(
            (historical_points.shape[0] * len(self.domain_xe), historical_points.shape[1]))

        for i in range(historical_points.shape[0]):
            for j in range(len(self.domain_xe)):
                first_part = historical_points[i][0:self.x_domain]
                point = np.concatenate((first_part, np.array(self.domain_xe[j])))
                create_vector[(i-1)*len(self.domain_xe) + j, :] = point
        cov_2 = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, self.gp.data['points'], create_vector, historical_points.shape[1])
        q23 = np.dot(
            cov_2, np.kron(np.identity(n), self.weights.reshape((len(self.weights),1))))

        cov_3 = self.gp.evaluate_cov(create_vector, parameters_kernel)
        q33 = np.dot(np.kron(np.identity(n), self.weights.reshape((1, len(self.weights)))), cov_3)
        q33 = np.dot(q33, np.kron(np.identity(n), self.weights.reshape((len(self.weights),1))))

        C = np.concatenate((cov_1, q23), axis=1)
        c_aux = np.concatenate((q23.transpose(), q33), axis=1)
        C= np.concatenate((C, c_aux), axis=0)

        chol = cholesky(C, max_tries=7)

        solv = cho_solve(chol, Z.reshape((len(Z), 1)))
        solv_2 = cho_solve(chol, one.reshape((len(one), 1)))

        bc = np.dot(one, solv) / np.dot(one, solv_2)

        candidate_vector = np.zeros((len(self.domain_xe), historical_points.shape[1]))
        for j in range(len(self.domain_xe)):
            point = np.concatenate((candidate_point, np.array(self.domain_xe[j])))
            candidate_vector[j, :] = point
        cov_2 = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, candidate_vector, create_vector, historical_points.shape[1])

        cov_4 = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, candidate_vector, historical_points, historical_points.shape[1])

        weights_matrix = self.weights.reshape((len(self.weights),1))
        kron = np.kron(np.identity(n), weights_matrix)
        big_cov = np.concatenate((cov_4, np.dot(cov_2, kron)), axis=1)

        c = np.dot(weights_matrix.transpose(), big_cov)

        vec_1 = Z - bc * one
        part_1 = np.dot(c, cho_solve(chol, vec_1.reshape((len(vec_1), 1))))
        mc = bc + part_1
        return mc[0,0], c, chol, Z, bc

    def ei_given_sample(self, sample, parameters_kernel, candidate_point):
        """
        Correct
        See p. 1140. We compute (8)
        :param sample:
        :param parameters_kernel:
        :param candidate_point:
        :return:
        """
        M = np.min(sample)
        n = len(sample)
        mc,c,chol, Z, bc = self.compute_mc_given_sample(sample, candidate_point, parameters_kernel)
        historical_points = self.gp.data['points']

        candidate_vector = np.zeros((len(self.domain_xe), historical_points.shape[1]))
        for j in range(len(self.domain_xe)):
            point = np.concatenate((candidate_point, np.array(self.domain_xe[j])))
            candidate_vector[j, :] = point
        cov_new = self.gp.evaluate_cov(candidate_vector, parameters_kernel)
        weights_matrix = self.weights.reshape((len(self.weights), 1))

        Rc = np.dot(weights_matrix.transpose(), np.dot(cov_new, weights_matrix))
        Rc -= np.dot(c, cho_solve(chol, c.transpose()))
        one = np.ones((2 * n, 1))
        Rc += (1 - np.dot(c, cho_solve(chol, one))) ** 2 / \
              (np.dot(one.transpose(), cho_solve(chol, one)))

        Zc = Z.reshape((len(Z),1))
        sigma_c = np.dot(Zc.transpose(), cho_solve(chol, Zc))
        sigma_c -= (bc ** 2) * np.dot(one.transpose(), cho_solve(chol, one))
        sigma_c /= (2.0 * n - 1)

        difference = M - mc
        sd = 1.0 / np.sqrt(Rc * sigma_c)
        component_1 = (M - mc) * t.cdf(difference * sd, 2 * n - 1)

        component_2 = t.pdf(difference * sd, 2 * n - 1)
        component_2 *= 1.0 / (2.0 * (n - 1))
        component_2 *= (2.0 * n - 1) * np.sqrt(Rc * sigma_c) + (difference ** 2) * sd

        result = component_1 + component_2

        return result[0, 0]


    def ei_objective(self, x, samples, parameters_kernel):
        """
        Correct
        :param x:
        :param samples:
        :param parameters_kernel:
        :return:
        """
        values = []
        for sample in samples:
            value = self.ei_given_sample(sample, parameters_kernel, x)
            values.append(value)
        return np.mean(values)

    def evaluate_squared_error(self, environment, control, parameters_kernel):
        """
        Correct
        :param control:
        :param environment:
        :param parameters_kernel:
        :return:
        """
        new_point = np.concatenate((control, environment))
        new_point = new_point.reshape((1, len(new_point)))

        historical_points = self.gp.data['points']
        dim_kernel = historical_points.shape[1]
        cov = self.gp.evaluate_cov(historical_points, parameters_kernel)
        chol = cholesky(cov, max_tries=7)

        var, beta = self.estimate_variance_gp(parameters_kernel)

        y = self.gp.data['evaluations']
        n = len(y)

        r = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, historical_points, new_point, dim_kernel)
        one = np.ones(n)
        vect = y - beta * one
        m1 = beta + np.dot(r.transpose(), cho_solve(chol, vect.reshape((len(vect), 1))))
        M = np.concatenate((y,  m1[0, :]))
        M = M.reshape((len(M), 1))

        candidate_vector = np.zeros((len(self.domain_xe), dim_kernel))
        for j in range(len(self.domain_xe)):
            point = np.concatenate((control, np.array(self.domain_xe[j])))
            candidate_vector[j, :] = point
        R3 = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, candidate_vector, candidate_vector, dim_kernel)
        R3SN = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, candidate_vector, historical_points, dim_kernel)
        weights_matrix = self.weights.reshape((len(self.weights), 1))
        r3 = self.gp.kernel.evaluate_cross_cov_defined_by_params(
            parameters_kernel, candidate_vector, new_point, dim_kernel)
        R = np.dot(weights_matrix.transpose(), np.dot(R3, weights_matrix))

        e12 = np.dot(weights_matrix.transpose(), np.concatenate((R3SN, r3), axis=1))

        covE = np.concatenate((cov, r), axis=1)
        one_matrix = np.ones((r.shape[1], r.shape[1]))
        cov_aux = np.concatenate((r.transpose(), one_matrix), axis=1)
        covE = np.concatenate((covE, cov_aux), axis=0)

        cholE = cholesky(covE, max_tries=7)
        R -= np.dot(e12, cho_solve(cholE, e12.transpose()))

        temp = (1 - np.dot(e12, cho_solve(cholE, np.ones((n+1, 1))))) ** 2
        temp /= np.dot(np.ones((n+1, 1)).transpose(), cho_solve(cholE, np.ones((n+1, 1))))
        R += temp

        aux = np.dot(M.transpose(), cho_solve(cholE, M))
        ones_vec = np.ones((n+1,1))
        aux_2 = np.dot(np.dot(M.transpose(), cho_solve(cholE, ones_vec)), ones_vec.transpose())
        aux_2 = np.dot(aux_2, cho_solve(cholE, M))
        aux_2 /= (np.dot(ones_vec.transpose(), cho_solve(cholE, ones_vec)))
        aux -= aux_2
        aux += ((n - 1) / float(n - 3)) * var

        value = aux
        value *= 1.0 / (n - 2)
        value *= R
        return value

    def get_environment(self, control, parameters_kernel, n_restarts=10):
        """
        correct
        See p.1142, eq. 15
        :param control:
        :return:

        """
        bounds = [tuple(bound) for bound in [self.gp.bounds[i] for i in self.w_domain]]
        bounds_2 = []
        for bound in bounds:
            bounds_2.append([bound[0], bound[-1]])
        bounds = bounds_2
        start = DomainService.get_points_domain(
            n_restarts, bounds)

        dim = len(start[0])
        start_points = {}
        for i in range(n_restarts):
            start_points[i] = start[i]
        optimization = Optimization(
            NELDER,
            wrapper_evaluate_squared_error,
            bounds,
            None,
            hessian=None, tol=None,
            minimize=False)

        args = (False, None, True, 0, optimization, self, control, parameters_kernel)
        sol = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, start_points, *args)
        solutions = []
        results_opt = []
        for i in range(n_restarts):
            if sol.get(i) is None:
                logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                            % i)
                continue
            solutions.append(sol.get(i)['optimal_value'])
            results_opt.append(sol.get(i))
        ind_max = np.argmax(solutions)
        environment = results_opt[ind_max]['solution']

        return environment

    def estimate_parameters_kernel(self, n_restarts=10):
        """
        Correct
        :param n_restarts:
        :return:
        """
        start = self.gp.sample_parameters_posterior(n_restarts)

        start = [sample[2:] for sample in start]
        dim = len(start[0])
        start_points = {}
        for i in xrange(n_restarts):
            start_points[i] = start[i]
        optimization = Optimization(
            NELDER,
            wrapper_log_posterior_distribution_length_scale,
            [(None, None) for i in range(dim)],
            None,
            hessian=None, tol=None,
            minimize=False)
        args = (False, None, True, 0, optimization, self)
        sol = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, start_points, *args)
        solutions = []
        results_opt = []
        for i in xrange(n_restarts):
            if sol.get(i) is None:
                logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                            % i)
                continue
            solutions.append(sol.get(i)['optimal_value'])
            results_opt.append(sol.get(i))
        ind_max = np.argmax(solutions)

        self.parameters = results_opt[ind_max]['solution']

        return results_opt[ind_max]['solution']

    def iteration_algorithm(self, n_restarts=10, n_samples=10):
        """
        Checked
        :param n_restarts:
        :param n_samples:
        :return:
        """
        if self.parameters is None:
            self.estimate_parameters_kernel()
            parameters = self.parameters
        else:
            parameters = self.parameters

        samples = self.sample_variable(parameters, n_samples)

        bounds = [tuple(bound) for bound in [self.gp.bounds[i] for i in range(self.x_domain)]]
        start = DomainService.get_points_domain(
            n_restarts, self.gp.bounds[0:self.x_domain], type_bounds=self.gp.type_bounds[0:self.x_domain])

        dim = len(start[0])
        start_points = {}
        for i in range(n_restarts):
            start_points[i] = start[i]
        optimization = Optimization(
            NELDER,
            wrapper_ei_objective,
            bounds,
            None,
            hessian=None, tol=None,
            minimize=False)
        args = (False, None, True, 0, optimization, self, samples, parameters)
        sol = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, start_points, *args)
        solutions = []
        results_opt = []
        for i in range(n_restarts):
            if sol.get(i) is None:
                logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                            % i)
                continue
            solutions.append(sol.get(i)['optimal_value'])
            results_opt.append(sol.get(i))
        ind_max = np.argmax(solutions)
        control = results_opt[ind_max]['solution']
        # do in parallel

        environment = self.get_environment(control, parameters)

        return np.concatenate((control, environment))

    def optimize_mean(self, n_restarts=10):
        """
        Checked
        :param n_restarts:
        :return:
        """
        if self.parameters is None:
            self.estimate_parameters_kernel()
            parameters = self.parameters
        else:
            parameters = self.parameters
        bounds = [tuple(bound) for bound in [self.gp.bounds[i] for i in range(self.x_domain)]]
        start = DomainService.get_points_domain(
            n_restarts, self.gp.bounds[0:self.x_domain], type_bounds=self.gp.type_bounds[0:self.x_domain])
        dim = len(start[0])
        start_points = {}
        for i in range(n_restarts):
            start_points[i] = start[i]
        optimization = Optimization(
            NELDER,
            wrapper_mean_objective,
            bounds,
            None,
            hessian=None, tol=None,
            minimize=False)
        args = (False, None, True, 0, optimization, self, parameters)
        sol = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, start_points, *args)
        solutions = []
        results_opt = []
        for i in range(n_restarts):
            if sol.get(i) is None:
                logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                            % i)
                continue
            solutions.append(sol.get(i)['optimal_value'])
            results_opt.append(sol.get(i))
        ind_max = np.argmax(solutions)

        sol = results_opt[ind_max]
        sol['optimal_value'] = [sol['optimal_value']]

        return sol


    def mean_objective(self, candidate_point, parameters_kernel):
        """
        checked
        :param x:
        :return:
        """
        sample = self.compute_expectation_sample(parameters_kernel)
        return self.compute_mc_given_sample(sample, candidate_point, parameters_kernel)[0]

    def optimize(self, n_restarts=10, n_samples=10, **kwargs):
        return {'solution': self.iteration_algorithm(n_restarts, n_samples), 'optimal_value': 0}


    def clean_cache(self):
        """
        Cleans the cache
        """
        self.gp.clean_cache()
        self.parameters = None