from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
import math

from os import path
import os

from copy import deepcopy

import multiprocessing as mp

import sys

import itertools

import numpy as np
from scipy.linalg import lapack
from scipy import linalg

from stratified_bayesian_optimization.lib.constant import (
    UNIFORM_FINITE,
    LBFGS_NAME,
    DEBUGGING_DIR,
    TASKS,
    SGD_NAME,
    NEWTON_CG_NAME,
    DOGLEG,
)
from stratified_bayesian_optimization.bayesian.bayesian_evaluations import BayesianEvaluations
from stratified_bayesian_optimization.lib.affine_break_points import (
    AffineBreakPointsPrep,
    AffineBreakPoints,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.lib.util import (
    wrapper_optimization,
    wrapper_objective_voi,
    wrapper_gradient_voi,
    wrapper_evaluate_sbo_by_sample,
    wrapper_optimize,
    wrapper_evaluate_sample,
    wrapper_evaluate_gradient_sample,
    wrapper_evaluate_sbo_mc,
    wrapper_get_parameters_for_samples,
    wrapper_evaluate_sample_bayesian,
    wrapper_evaluate_sbo_by_sample_bayesian,
    wrapper_evaluate_sbo_by_sample_2,
    wrapper_grad_voi_sgd,
    wrapper_sgd,
    wrapper_get_parameters_for_samples_2,
    wrapper_evaluate_sbo_by_sample_bayesian_2,
    wrapper_evaluate_hessian_sample,
    wrapper_evaluate_sbo_by_sample_no_sp,
    wrapper_optimize_posterior_mean,
)
from stratified_bayesian_optimization.lib.constant import DEFAULT_N_PARAMETERS, DEFAULT_N_SAMPLES
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.util import wrapper_evaluate_sbo
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.acquisition_functions.multi_task import MultiTasks
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature


logger = SBOLog(__name__)


class SBO(object):

    _filename = 'opt_sbo_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}_mc_{monte_carlo}_samples_params_' \
                '{n_samples_parameters}_sbo.json'.format

    _filename_voi_evaluations = '{iteration}_sbo_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}_mc_' \
                                '{monte_carlo}.json'.format

    _filename_points_voi_evaluations = 'points_for_voi_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format


    def __init__(self, bayesian_quadrature, discretization_domain=None):
        """

        :param bayesian_quadrature: a bayesian quadrature instance.
        :param discretization_domain: np.array(mxl), discretization of the domain of x.
        """

        self.bq = bayesian_quadrature
        self.discretization = discretization_domain

        self.bounds_opt = deepcopy(self.bq.bounds)
        if self.bq.separate_tasks and not self.bq.task_continue:
            self.bounds_opt.append([None, None])
        elif self.bq.separate_tasks:
            for i in self.bq.w_domain:
                bound = self.bounds_opt[i]
                self.bounds_opt[i] = [bound[0], bound[-1]]
        self.opt_separing_domain = False

        # Bounds or list of number of points of the domain
        self.domain_w = [self.bq.gp.bounds[i] for i in self.bq.w_domain]

        if self.bq.distribution == UNIFORM_FINITE:
            self.opt_separing_domain = True

        self.optimization_results = []

        # Cached data for MC SBO
        self.samples = None # Samples from a standard Gaussian r.v. used to estimate SBO
        self.starting_points_sbo = None

        # 'optimum': arg_max{a_n(x) + sigma(x, candidate_point)*sample}
        # 'max': max{a_n(x) + sigma(x, candidate_point)*sample}
        self.optimal_samples = {}

        # Cached evaluations of the SBO by taking samples of the parameters of the model.
        self.mc_bayesian = {}
        self.args_handler = ()


    def add_file_to_log(self, model_type, problem_name, training_name, n_training, random_seed,
                        n_samples_parameters):
        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        logger.add_file_to_log(model_type, problem_name, kernel_name, training_name, n_training,
                               random_seed, n_samples_parameters)

    def evaluate_sample(self, point, candidate_point, sample, var_noise=None, mean=None,
                        parameters_kernel=None, cache=True, n_threads=0, clear_cache=True):
        """
        Evaluate a sample of a_{n+1}(point) given that candidate_point is chosen.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param sample: (float) a sample from a standard Gaussian r.v.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param n_threads: (int)
        :return: float
        """

        if len(point.shape) == 1:
            point = point.reshape((1, len(point)))

        vectors = self.bq.compute_parameters_for_sample(
            point, candidate_point, var_noise=var_noise, mean=mean,
            parameters_kernel=parameters_kernel, cache=cache, n_threads=n_threads,
            clear_cache=clear_cache)
        value = vectors['a'] + sample * vectors['b']

        return value[0, 0]

    def evaluate_gradient_sample(self, point, candidate_point, sample, var_noise=None, mean=None,
                        parameters_kernel=None, cache=True):
        """
        Evaluate the gradient of a sample of a_{n+1}(point) given that candidate_point is chosen.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param sample: (float) a sample from a standard Gaussian r.v.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :return: np.array(n)
        """

        if len(point.shape) == 1:
            point = point.reshape((1, len(point)))

        gradient_params = self.bq.compute_gradient_parameters_for_sample(
            point, candidate_point, var_noise=var_noise, mean=mean,
            parameters_kernel=parameters_kernel, cache=cache
        )

        grad_a = gradient_params['a']
        grad_b = gradient_params['b']
        grad = grad_a + sample * grad_b

        return grad

    def evaluate_hessian_sample(self, point, candidate_point, sample, var_noise=None, mean=None,
                        parameters_kernel=None, cache=True):
        """
        Evaluate the hessian of a sample of a_{n+1}(point) given that candidate_point is chosen.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param sample: (float) a sample from a standard Gaussian r.v.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :return: np.array(nxn)
        """

        if len(point.shape) == 1:
            point = point.reshape((1, len(point)))

        hessian_params = self.bq.compute_hessian_parameters_for_sample(
            point, candidate_point, var_noise=var_noise, mean=mean,
            parameters_kernel=parameters_kernel, cache=cache
        )

        hessian_a = hessian_params['a']
        hessian_b = hessian_params['b']
        hessian = hessian_a + sample * hessian_b

        diag_cov = np.diag(hessian)
        if np.any(diag_cov < 0.) and np.min(diag_cov) > -1e-6:
            max_tries = 6
            n_tries = 0

            jitter = min(np.mean(diag_cov[np.where(diag_cov < 0)]) * 1e-6, 1e-6)
            while np.any(diag_cov < 0.) and n_tries < max_tries and np.isfinite(jitter):
                hessian += np.eye(hessian.shape[0]) * jitter
                diag_cov = np.diag(hessian)
                n_tries += 1
                jitter *= 10

        return hessian

    def evaluate_sbo_by_sample(self, candidate_point, sample, start=None,
                               var_noise=None, mean=None, parameters_kernel=None, n_restarts=5,
                               parallel=True, n_threads=0, method_opt=None, tol=None,
                               **opt_params_mc):
        """
        Optimize a_{n+1}(x)  given the candidate_point and the sample of the Gaussian r.v.

        :param candidate_point: np.array(1xn)
        :param sample: float
        :param start: np.array(1xm)
        :param n_restarts: (int) Number of restarts of the optimization algorithm.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param parallel: (boolean) Multi-start optimization in parallel if it's True
        :param n_threads: (int)
        :param method_opt: [LBFGS_NAME, NEWTON_CG_NAME, TRUST_N_CG, DOGLEG]
        :param tol: float
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'max': float, 'optimum': np.array(n)}
        """

        bounds_x = [self.bounds_opt[i] for i in xrange(len(self.bounds_opt)) if i in
                    self.bq.x_domain]

        dim_x = len(bounds_x)
        vertex = None
        if dim_x < 7:
            vertex = []
            for point in itertools.product(*bounds_x):
                vertex.append(point)

        if var_noise is None:
            index_cache = 'mc_mean'
        else:
            index_cache = (var_noise, mean, tuple(parameters_kernel))

        if start is None:
            start_points = DomainService.get_points_domain(n_restarts + 1, bounds_x,
                                                        type_bounds=len(bounds_x) * [0])
            if index_cache in self.bq.optimal_solutions and \
                            len(self.bq.optimal_solutions[index_cache]) > 0:
                start = self.bq.optimal_solutions[index_cache][-1]['solution']

                start = [start] + start_points[0 : -1]
                start = np.array(start)
            else:
                start = np.array(start_points)
        else:
            n_restarts = 0
            parallel = False

        bounds_x = [tuple(bound) for bound in bounds_x]

        if parallel:
            grad_function = wrapper_evaluate_gradient_sample
            objective_function = wrapper_evaluate_sample
            hessian_function = wrapper_evaluate_hessian_sample
        else:
            objective_function = self.evaluate_sample
            grad_function = self.evaluate_gradient_sample
            hessian_function = self.evaluate_hessian_sample

        if method_opt is None:
            method_opt = LBFGS_NAME

        optimization = Optimization(
            method_opt,
            objective_function,
            bounds_x,
            grad_function,
            hessian=hessian_function, tol=tol,
            minimize=False, **opt_params_mc)

        if parallel:
            solutions = []
            results_opt = []
            point_dict = {}
            for i in xrange(n_restarts + 1):
                point_dict[i] = start[i, :]

            args = (False, None, True, n_threads, optimization, self, candidate_point, sample,
                    var_noise, mean, parameters_kernel)

            sol = Parallel.run_function_different_arguments_parallel(
                wrapper_optimize, point_dict, *args)

            for i in xrange(n_restarts + 1):
                if sol.get(i) is None:
                    logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                                % i)
                    continue
                solutions.append(sol.get(i)['optimal_value'])
                results_opt.append(sol.get(i))
        else:
            solutions = []
            results_opt = []
            for i in xrange(n_restarts + 1):
                start_ = start[i, :]
                args = (candidate_point, sample, var_noise, mean, parameters_kernel)
                results = optimization.optimize(start_, *args)
                results_opt.append(results)
                solutions.append(results['optimal_value'])

        max_ = np.max(solutions)
        arg_max = results_opt[np.argmax(solutions)]['solution']

        if vertex is not None:
            n = len(vertex)
            point_dict = {}
            args = (False, None, parallel, 0, self, candidate_point, sample, var_noise, mean,
                    parameters_kernel)

            for j in range(n):
                point_dict[j] = np.array(vertex[j])
            values = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sample, point_dict, *args)

            values_candidates = []
            for j in range(n):
                values_candidates.append(values[j])
            ind_max_2 = np.argmax(values_candidates)

            if np.max(values_candidates) > max_:
                max_ = np.max(values_candidates)
                arg_max = point_dict[ind_max_2]

        return {'max': max_, 'optimum': arg_max}

    def generate_samples_starting_points_evaluate_mc(self, n_samples, n_restarts, cache=True):

        samples = np.random.normal(0, 1, n_samples)

        if cache:
            self.samples = samples
        start = self.generate_starting_points_evaluate_mc(n_restarts, cache=cache)

        return samples, start

    def generate_starting_points_evaluate_mc(self, n_restarts, cache=True):
        bounds_x = [self.bounds_opt[i] for i in xrange(len(self.bounds_opt)) if i in
                 self.bq.x_domain]

        start_points = DomainService.get_points_domain(n_restarts + 1, bounds_x,
                                                       type_bounds=len(bounds_x) * [0])
        prev_starts = []
        # if len(self.bq.optimal_solutions) > 0:
        #     for index in self.bq.optimal_solutions:
        #         if len(self.bq.optimal_solutions[index]) > 0:
        #             start = self.bq.optimal_solutions[index][-1]['solution']
        #             prev_starts.append(start)
        if len(prev_starts) > 0:
            start = prev_starts + start_points
            start = np.array(start)
        else:
            start = np.array(start_points)

        if cache:
            self.starting_points_sbo = start
        return start


    def evaluate_mc_bayesian(self, candidate_point, n_samples_parameters, n_samples,
                             n_restarts=10, n_best_restarts=0, n_threads=0, compute_max_mean=False,
                             method_opt=None, **opt_params_mc):
        """
        Evaluate SBO policy following a Bayesian approach.
        :param candidate_point:
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param compute_max_mean
        :param method_opt
        :param opt_params_mc:
        :return: float
        """
        parameters = self.bq.gp.samples_parameters[-n_samples_parameters:]


        if self.samples is not None:
            samples = self.samples
            start = self.starting_points_sbo
            n_restarts = start.shape[0]
        else:
            self.generate_samples_starting_points_evaluate_mc(n_samples, n_restarts)
            samples = self.samples
            start = self.starting_points_sbo
            n_restarts = start.shape[0]

        if tuple(candidate_point[0,:]) in self.mc_bayesian:
            value = self.mc_bayesian[tuple(candidate_point[0,:])]
            return value

        arguments = {}
        for i in xrange(n_samples_parameters):
            arguments[i] = [parameters[i][2:], parameters[i][0], parameters[i][1]]

        args = (False, None, True, n_threads, candidate_point, self)
        Parallel.run_function_different_arguments_parallel(
            wrapper_get_parameters_for_samples, arguments, *args)

        point_dict = {}
        point_start = {}
        max_values = []

        for k in xrange(n_samples_parameters):
            for i in xrange(n_samples):
                for j in xrange(n_restarts):
                    point_dict[(j, i, k)] = [deepcopy(start[j:j + 1, :]), samples[i], parameters[k]]
                    point_start[(j, i, k)] = [deepcopy(start[j:j + 1, :]), candidate_point,
                                           samples[i], parameters[k]]
        n_restarts_ = n_restarts
        if n_best_restarts > 0 and n_best_restarts < n_restarts:
            point_dict = {}

            args = (False, None, True, n_threads, self)
            values_candidates = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sample_bayesian, point_start, *args)

            for k in xrange(n_samples_parameters):
                for i in xrange(n_samples):
                    values = [values_candidates[(h, i, k)] for h in xrange(n_restarts)]
                    values_index = sorted(range(len(values)), key=lambda s: values[s])
                    values_index = values_index[-n_best_restarts:]

                    for j in xrange(len(values_index)):
                        index_p = values_index[j]
                        point_dict[(j, i, k)] = [point_start[(index_p, i, k)][0], samples[i],
                                                 parameters[k]]

            n_restarts_ = len(values_index)

        if method_opt is None:
            method_opt = LBFGS_NAME

        args = (False, None, True, n_threads, self, candidate_point, n_threads, method_opt)

        simulated_values = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_sbo_by_sample_bayesian, point_dict, *args, **opt_params_mc)

        values_parameters = []
        self.optimal_samples = {}

        for k in xrange(n_samples_parameters):
            index_cache_2 = (tuple(candidate_point[0, :]), tuple(parameters[k]))
            self.optimal_samples[index_cache_2] = {}
            self.optimal_samples[index_cache_2]['optimum'] = {}
            max_values = []
            for i in xrange(n_samples):
                values = []
                for j in xrange(n_restarts_):
                    if simulated_values.get((j, i, k)) is None:
                        logger.info("Error in computing simulated value at sample %d" % i)
                        continue
                    values.append(simulated_values[(j, i, k)]['max'])
                maximum = simulated_values[(np.argmax(values), i, k)]['optimum']
                self.optimal_samples[index_cache_2]['optimum'][i] = maximum
                max_ = np.max(values)
                max_values.append(max_)

            params = parameters[k]
            index_cache = (params[0], params[1], tuple(params[2:]))

            if not compute_max_mean:
                max_mean = 0

                if index_cache in self.bq.max_mean:
                    max_mean = self.bq.max_mean[index_cache]
            else:
                if index_cache not in self.bq.max_mean:
                    self.bq.optimize_posterior_mean(n_treads=0, var_noise=params[0],
                        parameters_kernel=params[2:], mean=params[1], n_best_restarts=100)
                max_mean = self.bq.max_mean[index_cache]

            values_parameters.append(np.mean(max_values) - max_mean)
        sbo_value = np.mean(values_parameters)

        self.mc_bayesian = {}
        self.mc_bayesian[tuple(candidate_point[0,:])] = sbo_value

        return sbo_value

    def evaluate_mc_bayesian_candidate_points_no_restarts(
            self, candidate_points, n_samples_parameters, n_samples, n_restarts=10,
            n_threads=0, compute_max_mean=False, compute_gradient=False,
            method_opt=None, **opt_params_mc):
        """
        Computes the SBO in parallel for several candidate_points. We don't do in parallel the
        restart points.

        :param candidate_points: np.array(nxk)
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_threads:
        :param compute_max_mean:
        :param compute_gradient: boolean
        :param method_opt: str
        :param opt_params_mc:
        :return: {'evaluations': np.array(n), 'gradient': np.array(nxk)}
        """

        if n_samples_parameters == 0:
            var_noise = self.bq.gp.var_noise.value[0]
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array
            mean = self.bq.gp.mean.value[0]
            param = [var_noise, mean] + list(parameters_kernel)
            parameters = [np.array(param)]
            n_samples_parameters += 1
        else:
            parameters = self.bq.gp.samples_parameters[-n_samples_parameters:]

        samples, start = self.generate_samples_starting_points_evaluate_mc(
            n_samples, n_restarts, cache=False)

        n_restarts = start.shape[0]

        arguments = {}

        n_candidate_points = candidate_points.shape[0]

        for j in xrange(n_candidate_points):
            for i in xrange(n_samples_parameters):
                arguments[(j, i)] = [parameters[i][2:], parameters[i][0], parameters[i][1],
                                     candidate_points[j:j+1,:]]

        args = (False, None, True, n_threads, self)
        Parallel.run_function_different_arguments_parallel(
            wrapper_get_parameters_for_samples_2, arguments, *args)

        point_start = {}

        for l in xrange(n_candidate_points):
            for k in xrange(n_samples_parameters):
                for i in xrange(n_samples):
                    point_start[(i, k, l)] = \
                        [candidate_points[l:l+1,:], samples[i], parameters[k]]


        point_dict = point_start

        if method_opt is None:
            method_opt = LBFGS_NAME

        args = (False, None, True, n_threads, self, n_threads, method_opt, n_restarts)

        simulated_values = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_sbo_by_sample_no_sp, point_dict, *args, **opt_params_mc)

        evaluations = np.zeros(n_candidate_points)
        gradient = None

        if compute_gradient:
            gradient = np.zeros((n_candidate_points, candidate_points.shape[1]))
            samples = samples.reshape((n_samples, 1))

        if n_samples_parameters > 0 and compute_max_mean:
            parameters_dict = {}

            for i, parameter in enumerate(parameters):
                index_cache = (parameter[0], parameter[1], tuple(parameter[2:]))

                if index_cache not in self.bq.max_mean:
                    parameters_dict[i] = parameter

            if len(parameters_dict) > 0:
                args = (False, None, True, 0, self.bq, None, LBFGS_NAME, 100)

                sol = Parallel.run_function_different_arguments_parallel(
                    wrapper_optimize_posterior_mean, parameters_dict, *args)
                for i in sol:
                    opt = sol.get(i)
                    par = parameters_dict[i]
                    index_cache = (par[0], par[1], tuple(par[2:]))
                    self.bq.max_mean[index_cache] = opt['optimal_value']

                    if index_cache not in self.bq.optimal_solutions:
                        self.bq.optimal_solutions[index_cache] = []

                    self.bq.optimal_solutions[index_cache].append(opt)

        for l in xrange(n_candidate_points):
            gradients = []
            values_parameters = []
            gradient_nan = False
            candidate_point = candidate_points[l:l+1, :]
            for k in xrange(n_samples_parameters):
                optimum_values = np.zeros((n_samples, len(self.bq.x_domain)))
                max_values = []
                param = parameters[k]

                for i in xrange(n_samples):
                    max_values.append(simulated_values[(i, k, l)]['max'])
                    optimum_values[i, :] = simulated_values[(i, k, l)]['optimum']

                params = parameters[k]
                index_cache = (params[0], params[1], tuple(params[2:]))

                if not compute_max_mean:
                    max_mean = 0
                    if index_cache in self.bq.max_mean:
                        max_mean = self.bq.max_mean[index_cache]
                else:
                    max_mean = self.bq.max_mean[index_cache]

                values_parameters.append(np.mean(max_values) - max_mean)

                if compute_gradient and not gradient_nan:
                    gradient_b = self.bq.gradient_vector_b(
                        candidate_point, optimum_values, var_noise=param[0], mean=param[1],
                        parameters_kernel=param[2:], cache=True, parallel=True, monte_carlo=True,
                        n_threads=n_threads)

                    gradient_ = gradient_b * samples
                    gradient_approx = np.mean(gradient_, axis=0)
                    gradients.append(gradient_approx)

                    if gradient_b is np.nan:
                        gradient_nan = True
                        gradients = np.nan


            if compute_gradient:
                if gradient_nan:
                    gradient[l, :] = np.array(candidate_points.shape[1] * [np.nan])
                else:
                    gradient[l, :] = np.mean(gradients, axis=0)
            evaluations[l] = np.mean(values_parameters)

        return {'evaluations': evaluations, 'gradient': gradient}

    def evaluate_mc_bayesian_candidate_points(
            self, candidate_points, n_samples_parameters, n_samples, n_restarts=10,
            n_best_restarts=0, n_threads=0, compute_max_mean=False, compute_gradient=False,
            method_opt=None, **opt_params_mc):
        """
        Computes the SBO in parallel for several candidate_points
        :param candidate_points: np.array(nxk)
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param compute_max_mean:
        :param compute_gradient: boolean
        :param method_opt: str
        :param opt_params_mc:
        :return: {'evaluations': np.array(n), 'gradient': np.array(nxk)}
        """

        if n_samples_parameters == 0:
            var_noise = self.bq.gp.var_noise.value[0]
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array
            mean = self.bq.gp.mean.value[0]
            param = [var_noise, mean] + list(parameters_kernel)
            parameters = [np.array(param)]
            n_samples_parameters += 1
        else:
            parameters = self.bq.gp.samples_parameters[-n_samples_parameters:]

        samples, start = self.generate_samples_starting_points_evaluate_mc(
            n_samples, n_restarts, cache=False)

        n_restarts = start.shape[0]

        arguments = {}

        n_candidate_points = candidate_points.shape[0]

        for j in xrange(n_candidate_points):
            for i in xrange(n_samples_parameters):
                arguments[(j, i)] = [parameters[i][2:], parameters[i][0], parameters[i][1],
                                     candidate_points[j:j+1,:]]

        args = (False, None, True, n_threads, self)
        Parallel.run_function_different_arguments_parallel(
            wrapper_get_parameters_for_samples_2, arguments, *args)

        point_start = {}

        for l in xrange(n_candidate_points):
            for k in xrange(n_samples_parameters):
                for i in xrange(n_samples):
                    for j in xrange(n_restarts):
                        point_start[(j, i, k, l)] = \
                            [deepcopy(start[j:j + 1, :]), candidate_points[l:l+1,:], samples[i],
                             parameters[k]]

        n_restarts_ = n_restarts
        point_dict = {}
        if n_best_restarts > 0 and n_best_restarts < n_restarts:

            args = (False, None, True, n_threads, self)
            values_candidates = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sample_bayesian, point_start, *args)

            for l in xrange(n_candidate_points):
                for k in xrange(n_samples_parameters):
                    for i in xrange(n_samples):
                        values = [values_candidates[(h, i, k, l)] for h in xrange(n_restarts)]
                        values_index = sorted(range(len(values)), key=lambda s: values[s])
                        values_index = values_index[-n_best_restarts:]

                        for j in xrange(len(values_index)):
                            index_p = values_index[j]
                            point_dict[(j, i, k, l)] = \
                                [start[index_p : index_p + 1, :], candidate_points[l:l+1,:],
                                 samples[i], parameters[k]]
            n_restarts_ = len(values_index)
        else:
            point_dict = point_start

        if method_opt is None:
            method_opt = LBFGS_NAME

        args = (False, None, True, n_threads, self, n_threads, method_opt)

        simulated_values = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_sbo_by_sample_bayesian_2, point_dict, *args, **opt_params_mc)

        evaluations = np.zeros(n_candidate_points)
        gradient = None

        if compute_gradient:
            gradient = np.zeros((n_candidate_points, candidate_points.shape[1]))
            samples = samples.reshape((n_samples, 1))

        for l in xrange(n_candidate_points):
            gradients = []
            values_parameters = []
            candidate_point = candidate_points[l:l+1, :]
            for k in xrange(n_samples_parameters):
                optimum_values = np.zeros((n_samples, len(self.bq.x_domain)))
                max_values = []
                param = parameters[k]
                for i in xrange(n_samples):
                    values = []
                    for j in xrange(n_restarts_):
                        if simulated_values.get((j, i, k, l)) is None:
                            logger.info("Error in computing simulated value at sample %d" % i)
                            continue
                        values.append(simulated_values[(j, i, k, l)]['max'])
                    maximum = simulated_values[(np.argmax(values), i, k, l)]['optimum']
                    optimum_values[i, :] = maximum

                    max_ = np.max(values)
                    max_values.append(max_)

                params = parameters[k]
                index_cache = (params[0], params[1], tuple(params[2:]))

                if not compute_max_mean:
                    max_mean = 0
                    if index_cache in self.bq.max_mean:
                        max_mean = self.bq.max_mean[index_cache]
                else:
                    if index_cache not in self.bq.max_mean:
                        self.bq.optimize_posterior_mean(n_treads=0, var_noise=params[0],
                            parameters_kernel=params[2:], mean=params[1], n_best_restarts=100)
                    max_mean = self.bq.max_mean[index_cache]

                values_parameters.append(np.mean(max_values) - max_mean)

                if compute_gradient:
                    gradient_b = self.bq.gradient_vector_b(
                        candidate_point, optimum_values, var_noise=param[0], mean=param[1],
                        parameters_kernel=param[2:], cache=True, parallel=True, monte_carlo=True,
                        n_threads=n_threads)
                    gradient_ = gradient_b * samples
                    gradient_approx = np.mean(gradient_, axis=0)
                    gradients.append(gradient_approx)
            if compute_gradient:
                gradient[l, :] = np.mean(gradients, axis=0)
            evaluations[l] = np.mean(values_parameters)

        return {'evaluations': evaluations, 'gradient': gradient}

    def evaluate_mc_bayesian_candidate_points(
            self, candidate_points, n_samples_parameters, n_samples, n_restarts=10,
            n_best_restarts=0, n_threads=0, compute_max_mean=False, compute_gradient=False,
            method_opt=None, **opt_params_mc):
        """
        Computes the SBO in parallel for several candidate_points
        :param candidate_points: np.array(nxk)
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param compute_max_mean:
        :param compute_gradient: boolean
        :param method_opt: str
        :param opt_params_mc:
        :return: {'evaluations': np.array(n), 'gradient': np.array(nxk)}
        """

        if n_samples_parameters == 0:
            var_noise = self.bq.gp.var_noise.value[0]
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array
            mean = self.bq.gp.mean.value[0]
            param = [var_noise, mean] + list(parameters_kernel)
            parameters = [np.array(param)]
            n_samples_parameters += 1
        else:
            parameters = self.bq.gp.samples_parameters[-n_samples_parameters:]

        samples, start = self.generate_samples_starting_points_evaluate_mc(
            n_samples, n_restarts, cache=False)

        n_restarts = start.shape[0]

        arguments = {}

        n_candidate_points = candidate_points.shape[0]

        for j in xrange(n_candidate_points):
            for i in xrange(n_samples_parameters):
                arguments[(j, i)] = [parameters[i][2:], parameters[i][0], parameters[i][1],
                                     candidate_points[j:j+1,:]]

        args = (False, None, True, n_threads, self)
        Parallel.run_function_different_arguments_parallel(
            wrapper_get_parameters_for_samples_2, arguments, *args)

        point_start = {}

        for l in xrange(n_candidate_points):
            for k in xrange(n_samples_parameters):
                for i in xrange(n_samples):
                    for j in xrange(n_restarts):
                        point_start[(j, i, k, l)] = \
                            [deepcopy(start[j:j + 1, :]), candidate_points[l:l+1,:], samples[i],
                             parameters[k]]

        n_restarts_ = n_restarts
        point_dict = {}
        if n_best_restarts > 0 and n_best_restarts < n_restarts:

            args = (False, None, True, n_threads, self)
            values_candidates = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sample_bayesian, point_start, *args)

            for l in xrange(n_candidate_points):
                for k in xrange(n_samples_parameters):
                    for i in xrange(n_samples):
                        values = [values_candidates[(h, i, k, l)] for h in xrange(n_restarts)]
                        values_index = sorted(range(len(values)), key=lambda s: values[s])
                        values_index = values_index[-n_best_restarts:]

                        for j in xrange(len(values_index)):
                            index_p = values_index[j]
                            point_dict[(j, i, k, l)] = \
                                [start[index_p : index_p + 1, :], candidate_points[l:l+1,:],
                                 samples[i], parameters[k]]
            n_restarts_ = len(values_index)
        else:
            point_dict = point_start

        if method_opt is None:
            method_opt = LBFGS_NAME

        args = (False, None, True, n_threads, self, n_threads, method_opt)

        simulated_values = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_sbo_by_sample_bayesian_2, point_dict, *args, **opt_params_mc)

        evaluations = np.zeros(n_candidate_points)
        gradient = None

        if compute_gradient:
            gradient = np.zeros((n_candidate_points, candidate_points.shape[1]))
            samples = samples.reshape((n_samples, 1))

        for l in xrange(n_candidate_points):
            gradients = []
            values_parameters = []
            candidate_point = candidate_points[l:l+1, :]
            for k in xrange(n_samples_parameters):
                optimum_values = np.zeros((n_samples, len(self.bq.x_domain)))
                max_values = []
                param = parameters[k]
                for i in xrange(n_samples):
                    values = []
                    for j in xrange(n_restarts_):
                        if simulated_values.get((j, i, k, l)) is None:
                            logger.info("Error in computing simulated value at sample %d" % i)
                            continue
                        values.append(simulated_values[(j, i, k, l)]['max'])
                    maximum = simulated_values[(np.argmax(values), i, k, l)]['optimum']
                    optimum_values[i, :] = maximum

                    max_ = np.max(values)
                    max_values.append(max_)

                params = parameters[k]
                index_cache = (params[0], params[1], tuple(params[2:]))

                if not compute_max_mean:
                    max_mean = 0
                    if index_cache in self.bq.max_mean:
                        max_mean = self.bq.max_mean[index_cache]
                else:
                    if index_cache not in self.bq.max_mean:
                        self.bq.optimize_posterior_mean(n_treads=0, var_noise=params[0],
                            parameters_kernel=params[2:], mean=params[1], n_best_restarts=100)
                    max_mean = self.bq.max_mean[index_cache]

                values_parameters.append(np.mean(max_values) - max_mean)

                if compute_gradient:
                    gradient_b = self.bq.gradient_vector_b(
                        candidate_point, optimum_values, var_noise=param[0], mean=param[1],
                        parameters_kernel=param[2:], cache=True, parallel=True, monte_carlo=True,
                        n_threads=n_threads)
                    gradient_ = gradient_b * samples
                    gradient_approx = np.mean(gradient_, axis=0)
                    gradients.append(gradient_approx)
            if compute_gradient:
                gradient[l, :] = np.mean(gradients, axis=0)
            evaluations[l] = np.mean(values_parameters)

        return {'evaluations': evaluations, 'gradient': gradient}

    def evaluate_gradient_given_sample_given_parameters(
            self, candidate_point, sample, parameters=None, n_restarts=10, n_best_restarts=0,
            n_threads=0, parallel=True, method_opt=None, **opt_params_mc):
        """
        Evaluate the gradient of E[max_{x}a_{n+1}(x)|candidate_point] given the sample and the
        parameters
        :param candidate_point: np.array(1xn)
        :param sample: float
        :param parameters: np.array(l)
        :param n_restarts: int
        :param n_best_restarts: int
        :param n_threads: int
        :param parallel: boolean
        :param opt_params_mc:
        :return: float
        """
        if parameters is None:
            var_noise = self.bq.gp.var_noise.value[0]
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array
            mean = self.bq.gp.mean.value[0]
        else:
            var_noise = parameters[0]
            mean = parameters[1]
            parameters_kernel = parameters[2:]

        if method_opt is None:
            method_opt = LBFGS_NAME

        if self.starting_points_sbo is None:
            self.generate_starting_points_evaluate_mc(n_restarts)

        start = self.starting_points_sbo
        n_restarts = start.shape[0]

        point_start = {}
        for i in xrange(n_restarts):
            point_start[i] = deepcopy(start[i:i + 1, :])

        if n_best_restarts > 0 and n_best_restarts < n_restarts:
            args = (False, None, parallel, n_threads, self, candidate_point, sample, var_noise, mean,
                    parameters_kernel, True, n_threads, True)
            values_candidates = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sample, point_start, *args)

            values = [values_candidates[i] for i in xrange(n_restarts)]
            values_index = sorted(range(len(values)), key=lambda s: values[s])
            values_index = values_index[-n_best_restarts:]

            point_start = {}
            for j in xrange(len(values_index)):
                index_p = values_index[j]
                point_start[j] = deepcopy(start[index_p:index_p + 1, :])

            n_restarts = len(values_index)

        args = (False, None, parallel, n_threads, self, sample, candidate_point, var_noise, mean,
                 parameters_kernel, 0, method_opt)

        values_opt = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_sbo_by_sample_2, point_start, *args, **opt_params_mc)

        values = []
        for j in xrange(n_restarts):
            if values_opt.get(j) is None:
                logger.info("Error in computing simulated value at sample %d" % i)
                continue
            values.append(values_opt[j]['max'])

        maximum = values_opt[np.argmax(values)]['optimum']

        maximum = maximum.reshape((1, len(maximum)))

        gradient_b = self.bq.gradient_vector_b(candidate_point, maximum, var_noise=var_noise,
                                               mean=mean, parameters_kernel=parameters_kernel,
                                               cache=True, parallel=False, monte_carlo=True,
                                               n_threads=0)
        if gradient_b is np.nan:
            return np.nan

        gradient = gradient_b * sample

        return gradient

    def evaluate_gradient_mc_bayesian(
            self, candidate_point, n_samples_parameters, n_samples, n_restarts=10,
            n_best_restarts=0, n_threads=0, compute_max_mean=False, method_opt=None,
            **opt_params_mc):
        """
        Evaluate the gradient of SBO by using MC estimation.

        :param candidate_point: np.array(1xn)
        :param parameters_kernel: np.array(l)
        :param var_noise: int
        :param mean: int
        :param n_samples: int
        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: int
        :parma n_best_restarts: int
        :param n_threads: int
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'gradient': np.array(n), 'std': np.array(n)}
        """


        if tuple(candidate_point[0,:]) not in self.mc_bayesian:
            self.evaluate_mc_bayesian(candidate_point, n_samples_parameters, n_samples,
                                 n_restarts=n_restarts, n_best_restarts=n_best_restarts,
                                      n_threads=n_threads, compute_max_mean=compute_max_mean,
                                      method_opt=method_opt, **opt_params_mc)

        gradients = []
        parameters = self.bq.gp.samples_parameters[-n_samples_parameters:]

        samples = self.samples.reshape((n_samples, 1))
        for param in parameters:
            index_cache_2 = (tuple(candidate_point[0, :]), tuple(param))
            max_points = self.optimal_samples[index_cache_2]['optimum']

            points = np.zeros((n_samples, len(self.bq.x_domain)))
            for i in xrange(n_samples):
                points[i, :] = max_points[i]

            gradient_b = self.bq.gradient_vector_b(candidate_point, points, var_noise=param[0],
                                                   mean=param[1], parameters_kernel=param[2:],
                                                   cache=True, parallel=True, monte_carlo=True,
                                                   n_threads=n_threads)
            if gradient_b is np.nan:
                return np.nan

            gradient = gradient_b * samples

            gradient_approx = np.mean(gradient, axis=0)
            gradients.append(gradient_approx)

        return np.mean(gradients, axis=0)

    def grad_voi_sgd(self, point, monte_carlo, bayesian, n_restarts=10, n_best_restarts=0,
                     n_threads=0, parallel=True, method_opt=None, random_seed=None,
                     **opt_params_mc):
        """
        Computes the objective voi using a bayesian approach. Used for the SGD. We useo only one
        sample of Z, and one sample of the parameters.

        :param point: np.array(n)
        :param monte_carlo:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param parallel:
        :param method_opt:
        :param random_seed:
        :param opt_params_mc:
        :return: float
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        point = point.reshape((1, len(point)))
        params = None
        sample = None
        if bayesian:
            params = self.bq.gp.sample_parameters(1)[0]
        if monte_carlo:
            sample = np.random.normal(0, 1, 1)[0]

        #TODO IMPLEMENT THE CASE WHEN MONTE_CARLO IS FALSE

        grad = self.evaluate_gradient_given_sample_given_parameters(
            point, sample, params, n_restarts, n_best_restarts, n_threads, parallel,
            method_opt=method_opt, **opt_params_mc)

        if grad is np.nan:
            return grad

        return grad[0, :]

    def objective_voi_bayesian(self, point, monte_carlo, n_samples_parameters, n_samples,
                               n_restarts, n_best_restarts, n_threads, method_opt=None,
                               **opt_params_mc):
        """
        Computes objective voi using a bayesian approach
        :param point:
        :param monte_carlo:
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param opt_params_mc:
        :return: float
        """

        if monte_carlo is False:
            args = (monte_carlo, n_samples, n_restarts, n_best_restarts, n_threads)
            value = BayesianEvaluations.evaluate(self.objective_voi, point, self.bq.gp,
                                               n_samples_parameters, None, *args,
                                               **opt_params_mc)[0]
        else:
            point = point.reshape((1, len(point)))
            value = self.evaluate_mc_bayesian(
                point, n_samples_parameters, n_samples, n_restarts,
                n_best_restarts, n_threads, method_opt=method_opt, **opt_params_mc)

        return value

    def grad_obj_voi_bayesian(self, point, monte_carlo, n_samples_parameters, n_samples,
                               n_restarts, n_best_restarts, n_threads, **opt_params_mc):
        """
        Computes gradient of voi using a Bayesian approach.
        :param point:
        :param monte_carlo:
        :param n_samples_parameters:
        :param n_samples:
        :param n_restarts:
        :param n_best_restarts:
        :param n_threads:
        :param opt_params_mc:
        :return:
        """
        if monte_carlo is False:
            args = (monte_carlo, n_samples, n_restarts, n_best_restarts, n_threads)
            value = BayesianEvaluations.evaluate(self.grad_obj_voi, point, self.bq.gp,
                                               n_samples_parameters, None, *args,
                                               **opt_params_mc)[0]
        else:
            point = point.reshape((1, len(point)))
            value = self.evaluate_gradient_mc_bayesian(
                point, n_samples_parameters, n_samples, n_restarts,
                n_best_restarts, n_threads, **opt_params_mc)

        return value

    def evaluate_mc(self, candidate_point,  n_samples, var_noise=None, mean=None,
                    parameters_kernel=None, random_seed=None, parallel=True, n_restarts=10,
                    n_best_restarts=0, n_threads=0, method_opt=None, **opt_params_mc):
        """
        Evaluate SBO policy by a MC estimation.

        :param candidate_point: np.array(1xn)
        :param n_samples: int
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: (int)
        :param n_best_restarts: (int)
        :param n_threads: (int)
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'value': float, 'std': float}
        """

        if method_opt is None:
            method_opt = LBFGS_NAME

        if random_seed is not None:
            np.random.seed(random_seed)

        if var_noise is None:
            var_noise = self.bq.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.bq.gp.mean.value[0]

        index_cache = (var_noise, mean, tuple(parameters_kernel))

        if index_cache in self.bq.max_mean:
            max_mean = self.bq.max_mean[index_cache]
        else:
            max_mean = self.bq.optimize_posterior_mean(
                random_seed=random_seed, n_treads=n_threads, var_noise=var_noise,
                parameters_kernel=parameters_kernel, mean=mean)['optimal_value']

        if self.samples is not None:
            samples = self.samples
            start = self.starting_points_sbo
            n_restarts = start.shape[0]
        else:
            self.generate_samples_starting_points_evaluate_mc(n_samples, n_restarts)
            samples = self.samples
            start = self.starting_points_sbo
            n_restarts = start.shape[0]

        max_values = []

        index_cache_2 = (tuple(candidate_point[0, :]), var_noise, mean, tuple(parameters_kernel))
        if index_cache_2 in self.optimal_samples:
            optimal_values = self.optimal_samples[index_cache_2]['max'].values()
            return {'value': np.mean(optimal_values) - max_mean,
                    'std': np.std(optimal_values) / n_samples}

        self.optimal_samples = {}
        self.optimal_samples[index_cache_2] = {}
        self.optimal_samples[index_cache_2]['max'] = {}
        self.optimal_samples[index_cache_2]['optimum'] = {}

        if parallel:
            # Cache this computation, so we don't have to do it over and over again
            self.bq.get_parameters_for_samples(True, candidate_point, parameters_kernel, var_noise,
                                               mean)

            point_dict = {}
            point_start = {}
            for i in xrange(n_samples):

                for j in xrange(n_restarts):
                    point_dict[(j, i)] = [deepcopy(start[j:j+1,:]), samples[i]]
                    point_start[(j, i)] = [deepcopy(start[j:j+1,:]), candidate_point, samples[i]]
            n_restarts_ = n_restarts

            if n_best_restarts > 0 and n_best_restarts < n_restarts:
                point_dict = {}
                args = (False, None, True, n_threads, self, var_noise, mean, parameters_kernel,
                        True, n_threads)
                values_candidates = Parallel.run_function_different_arguments_parallel(
                    wrapper_evaluate_sample, point_start, *args)

                for i in xrange(n_samples):
                    values = [values_candidates[(j, i)] for j in xrange(n_restarts)]
                    values_index = sorted(range(len(values)), key=lambda k: values[k])
                    values_index = values_index[-n_best_restarts:]

                    for j in xrange(len(values_index)):
                        index_p = values_index[j]
                        point_dict[(j, i)] = [point_start[(index_p, i)][0], samples[i]]
                n_restarts_ = len(values_index)

            args = (False, None, True, n_threads, self, candidate_point, var_noise, mean,
                    parameters_kernel, n_threads, method_opt)

            simulated_values = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sbo_by_sample, point_dict, *args, **opt_params_mc)
            for i in xrange(n_samples):
                values = []
                for j in xrange(n_restarts_):
                    if simulated_values.get((j, i)) is None:
                        logger.info("Error in computing simulated value at sample %d" % i)
                        continue
                    values.append(simulated_values[(j, i)]['max'])
                maximum = simulated_values[(np.argmax(values), i)]['optimum']
                max_ = np.max(values)
                self.optimal_samples[index_cache_2]['max'][i] = max_
                self.optimal_samples[index_cache_2]['optimum'][i] = maximum
                max_values.append(max_)
        else:
            for i in xrange(n_samples):
                max_value = self.evaluate_sbo_by_sample(
                    candidate_point, samples[i], start=None, var_noise=var_noise, mean=mean,
                    parameters_kernel=parameters_kernel, n_restarts=n_restarts, parallel=True,
                    method_opt=method_opt, tol=None, **opt_params_mc)
                max_values.append(max_value['max'])
                maximum = max_value['optimum']
                self.optimal_samples[index_cache_2]['max'][i] = max_value['max']
                self.optimal_samples[index_cache_2]['optimum'][i] = maximum

        return {'value': np.mean(max_values) - max_mean, 'std': np.std(max_values) / n_samples}

    def gradient_mc(self, candidate_point, var_noise=None, mean=None, parameters_kernel=None,
                    n_samples=None, random_seed=None, parallel=True, n_restarts=10,
                    n_best_restarts=0, n_threads=0, method_opt=None, **opt_params_mc):
        """
        Evaluate the gradient of SBO by using MC estimation.

        :param candidate_point: np.array(1xn)
        :param parameters_kernel: np.array(l)
        :param var_noise: int
        :param mean: int
        :param n_samples: int
        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: int
        :parma n_best_restarts: int
        :param n_threads: int
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'gradient': np.array(n), 'std': np.array(n)}
        """

        if method_opt is None:
            method_opt = LBFGS_NAME

        if var_noise is None:
            var_noise = self.bq.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.bq.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.bq.gp.mean.value[0]

        index_cache_2 = (tuple(candidate_point[0, :]), var_noise, mean, tuple(parameters_kernel))

        if index_cache_2 not in self.optimal_samples:
            self.evaluate_mc(candidate_point, n_samples, var_noise=var_noise, mean=mean,
                             parameters_kernel=parameters_kernel, random_seed=random_seed,
                             parallel=parallel, n_restarts=n_restarts,
                             n_best_restarts=n_best_restarts, n_threads=n_threads,
                             method_opt=method_opt, **opt_params_mc)

        max_points = self.optimal_samples[index_cache_2]['optimum']

        samples = self.samples
        n_samples = len(samples)

        points = np.zeros((n_samples, len(self.bq.x_domain)))
        for i in xrange(n_samples):
            points[i, :] = max_points[i]

        gradient_b = self.bq.gradient_vector_b(candidate_point, points, var_noise=var_noise,
                                               mean=mean, parameters_kernel=parameters_kernel,
                                               cache=True, parallel=parallel, monte_carlo=True,
                                               n_threads=n_threads)

        gradient = gradient_b * samples.reshape((n_samples, 1))

        gradient_approx = np.mean(gradient, axis=0)

        return {'gradient': gradient_approx, 'std': np.std(gradient, axis=0) / n_samples}


    def evaluate(self, point, var_noise=None, mean=None, parameters_kernel=None, cache=True,
                 n_threads=0):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param n_threads: (int) If n_threads > 0, it uses threads instead of processes.

        :return: float
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache, n_threads=n_threads)

        a = vectors['a']
        b = vectors['b']

        if not np.all(np.isfinite(b)):
            return 0.0

        a, b, keep = AffineBreakPointsPrep(a, b)

        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)

        return self.hvoi(b, c, keep1)


    def evaluate_gradient(self, point, var_noise=None, mean=None, parameters_kernel=None,
                          cache=True, n_threads=0):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param n_threads: int

        :return: np.array(n)
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache, n_threads=n_threads)

        a = vectors['a']
        b = vectors['b']

        a, b, keep = AffineBreakPointsPrep(a, b)
        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)
        M = len(keep1)

        if M <=1 :
            return np.zeros(point.shape[1])

        keep=keep[keep1] #indices conserved

        c=c[keep1+1]
        c2=np.abs(c[0:M-1])
        evalC=norm.pdf(c2)

        gradients = self.bq.gradient_vector_b(point, self.discretization[keep, :],
                                              var_noise=var_noise, mean=mean,
                                              parameters_kernel=parameters_kernel, cache=cache,
                                              keep_indexes=keep, n_threads=n_threads)

        gradient = np.zeros(point.shape[1])
        for i in xrange(point.shape[1]):
            gradient[i] = np.dot(np.diff(gradients[:, i]), evalC)

        return gradient

    def objective_voi(self, point, monte_carlo=False, n_samples=1, n_restarts=1, n_best_restarts=0,
                      n_threads=0, method_opt=None, *model_params, **opt_params_mc):
        """
        Evaluates the VOI at point.
        :param point: np.array(n)
        :param monte_carlo: (boolean) If True, estimates the function by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_best_restarts: (int)
        :param n_threads: (int)
        :param model_params: (var_noise (float), mean (float), parameters_kernel np.array(l))
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: float
        """

        if method_opt is None:
            method_opt = LBFGS_NAME

        point = point.reshape((1, len(point)))

        if not monte_carlo:
            value = self.evaluate(point, *model_params, n_threads=n_threads)
        else:
            value = self.evaluate_mc(point, n_samples, *model_params,
                                     n_restarts=n_restarts, n_best_restarts=n_best_restarts,
                                     n_threads=n_threads, method_opt=method_opt,
                                     **opt_params_mc)['value']

        return value

    def grad_obj_voi(self, point, monte_carlo=False, n_samples=1, n_restarts=1, n_best_restarts=0,
                     n_threads=0, method_opt=None, *model_params, **opt_params_mc):
        """
        Evaluates the gradient of VOI at point.
        :param point: np.array(n)
        :param monte_carlo: (boolean) If True, estimates the function by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_best_restarts: (int)
        :param n_threads: (int)
        :param model_params: (var_noise, mean, parameters_kernel)
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: np.array(n)
        """

        point = point.reshape((1, len(point)))

        if not monte_carlo:
            grad = self.evaluate_gradient(point, *model_params, n_threads=n_threads)
        else:
            grad = self.gradient_mc(point, *model_params, n_samples=n_samples,
                                    n_restarts=n_restarts, n_best_restarts=n_best_restarts,
                                    n_threads=n_threads, method_opt=method_opt,
                                    **opt_params_mc)['gradient']

        return grad

    def random_points_domain(self, n_points):
        """
        Sample n_points random points from the domain of SBO
        :param n_points: (int)
        :return: np.array(n_points x m)
        """
        bounds = self.bq.bounds

        if self.bq.separate_tasks:
            tasks = self.bq.tasks
            n_tasks = len(tasks)

            n_restarts = int(np.ceil(float(n_points) / n_tasks) * n_tasks)

            ind = [[i] for i in range(n_restarts)]
            np.random.shuffle(ind)
            task_chosen = np.zeros((n_restarts, 1))
            n_task_per_group = n_restarts / n_tasks

            for i in xrange(n_tasks):
                for j in xrange(n_task_per_group):
                    tk = ind[j + i * n_task_per_group]
                    task_chosen[tk, 0] = i

            start_points = DomainService.get_points_domain(
                n_restarts, bounds, type_bounds=self.bq.type_bounds)

            start_points = np.concatenate((start_points, task_chosen), axis=1)
        else:
            start_points = DomainService.get_points_domain(
                n_points, bounds, type_bounds=self.bq.type_bounds)

        return np.array(start_points)

    def optimize(self, start=None, random_seed=None, parallel=True, monte_carlo=False, n_samples=1,
                 n_restarts_mc=1, n_best_restarts_mc=0, n_restarts=1, n_best_restarts=0,
                 start_ei=True, n_samples_parameters=0, start_new_chain=True,
                 compute_max_mean_bayesian=False, maxepoch=10, default_n_samples=None,
                 default_n_samples_parameters=None, default_restarts_mc=None, method_opt_mc=None,
                 **opt_params_mc):
        """
        Optimizes the VOI.
        :param start: np.array(1xn)
        :param random_seed: int
        :param parallel: (boolean) For several tasks, it's run in paralle if it's True
        :param monte_carlo: (boolean) If True, estimates the objective function and gradient by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts_mc: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_best_restarts_mc: (int)
        :param n_restarts: (int)
        :param n_best_restarts: (int)
        :param start_ei: (boolean) If True, we choose starting points using EI.
        :param n_samples_parameters: (int) Number of samples of the parameters of the model. If
            n_samples_parameters = 0, we optimize SBO with the MLE parameters.
        :param start_new_chain: (boolen)
        :param compute_max_mean_bayesian: boolean
        :param maxepoch: (int) Max number of iterations in SGD
        :param default_n_samples: (int)
        :param default_n_samples_parameters: (int)
        :param default_restarts_mc: int
        :param method_opt_mc: str
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int

        :return: dictionary with the results of the optimization.
        """

        if method_opt_mc is None:
            method_opt_mc = LBFGS_NAME

        if random_seed is not None:
            np.random.seed(random_seed)

        if n_samples_parameters == 0:
            n_parameters = 0
        else:
            if default_n_samples_parameters is not None:
                n_parameters = default_n_samples_parameters
            else:
                n_parameters = DEFAULT_N_PARAMETERS

        if default_n_samples is None:
            default_n_samples = DEFAULT_N_SAMPLES

        if default_restarts_mc is None:
            default_restarts_mc = n_restarts_mc

        n_jobs = min(n_restarts, mp.cpu_count())
        n_threads = max(int((mp.cpu_count() - n_jobs) / n_jobs), 1)

        if n_samples_parameters > 0 and start_new_chain:
            self.bq.gp.start_new_chain()
            self.bq.gp.sample_parameters(n_parameters)
        elif n_samples_parameters > 0 and len(self.bq.gp.samples_parameters) < n_parameters:
            self.bq.gp.sample_parameters(n_parameters - len(self.bq.gp.samples_parameters))

        # if n_samples_parameters > 0 and compute_max_mean_bayesian:
        #     parameters = self.bq.gp.samples_parameters[-n_parameters:]
        #
        #     parameters_dict = {}
        #     for i, parameter in enumerate(parameters):
        #         index_cache = (parameter[0], parameter[1], tuple(parameter[2:]))
        #
        #         if index_cache not in self.bq.max_mean:
        #             parameters_dict[i] = parameter
        #
        #     if len(parameters_dict) > 0:
        #         args = (False, None, parallel, 0, self.bq, random_seed, LBFGS_NAME, 100)
        #
        #         sol = Parallel.run_function_different_arguments_parallel(
        #             wrapper_optimize_posterior_mean, parameters_dict, *args)
        #
        #         for i in sol:
        #             opt = sol.get(i)
        #             par = parameters_dict[i]
        #             index_cache = (par[0], par[1], tuple(par[2:]))
        #             self.bq.max_mean[index_cache] = opt['optimal_value']
        #
        #             if index_cache not in self.bq.optimal_solutions:
        #                 self.bq.optimal_solutions[index_cache] = []
        #
        #             self.bq.optimal_solutions[index_cache].append(opt)

        bounds = self.bq.bounds

        #TODO: CHANGE ei.optimize with n_samples_parameters
        points_st_ei = []
        values_ei = []
        if start_ei and not self.bq.separate_tasks:
            ei = EI(self.bq.gp)
            opt_ei = ei.optimize(n_restarts=100, n_samples_parameters=5,
                                 parallel=parallel, n_best_restarts=10, maxepoch=50)
            st_ei = opt_ei['solution']
            st_ei = st_ei.reshape((1, len(st_ei)))
            n_restarts -= 1
            n_best_restarts -= 1
        elif start_ei:
            quadrature_2 = BayesianQuadrature(self.bq.gp, self.bq.x_domain, self.bq.distribution,
                                            parameters_distribution=self.bq.parameters_distribution,
                                            model_only_x=True)
            mk = MultiTasks(quadrature_2, quadrature_2.parameters_distribution.get(TASKS))
            st_ei = mk.optimize(parallel=True, n_restarts=100,
                                n_best_restarts=10,
                                n_samples_parameters=5, maxepoch=50)
            st_ei = st_ei['solution']
            st_ei = st_ei.reshape((1, len(st_ei)))
            # n_restarts -= 1
            # n_best_restarts -= 1

        if start is None:
            if self.bq.separate_tasks and n_restarts > 0 and not self.bq.task_continue:
                tasks = self.bq.tasks
                n_tasks = len(tasks)

             #   n_restarts = int(np.ceil(float(n_restarts) / n_tasks) * n_tasks)
               # n_restarts = max(1, n_restarts / n_tasks)

                # ind = [[i] for i in range(n_restarts)]
                # np.random.shuffle(ind)
                # task_chosen = np.zeros((n_restarts, 1))
                # n_task_per_group = n_restarts / n_tasks
                #
                # for i in xrange(n_tasks):
                #     for j in xrange(n_task_per_group):
                #         tk = ind[j + i * n_task_per_group]
                #         task_chosen[tk, 0] = i

                new_points = DomainService.get_points_domain(
                    100, bounds, type_bounds=self.bq.type_bounds)
                current_points = np.array(self.bq.gp.data['points'])

                tasks_hist = current_points[:, -1]
                current_points = current_points[:, 0:-1]

                distances = Distances.dist_square_length_scale(
                    np.ones(len(new_points[0])), new_points, current_points)
                max_distances = np.min(distances, axis=1)

                sort_dist_ind = sorted(range(len(max_distances)), key=lambda k: max_distances[k])

                md = int(np.ceil(len(max_distances) / 2.0))
                uq = int(np.ceil(len(max_distances) / 4.0))
                lw = int(np.ceil(len(max_distances) / 8.0))

                index_1 = n_restarts / 2
                index_2 = n_restarts - index_1
                index_1 = np.random.choice(range(uq, md), index_1, replace=False)
                index_2 = np.random.choice(range(lw, uq), index_2, replace=False)

                index_1 = [sort_dist_ind[t] for t in index_1]
                index_2 = [sort_dist_ind[t] for t in index_2]
                index = index_1 + index_2

                start_points = []
                for i in index:
                    for j in range(n_tasks):
                        start_points.append(np.concatenate((new_points[i], [j])))
                start_points = np.array(start_points)
                # start_points = np.concatenate((start_points, task_chosen), axis=1)
            elif n_restarts > 0:
                new_points = DomainService.get_points_domain(
                    100, bounds, type_bounds=self.bq.type_bounds,
                    simplex_domain=self.bq.simplex_domain)
                current_points = np.array(self.bq.gp.data['points'])

                distances = Distances.dist_square_length_scale(
                    np.ones(len(new_points[0])), new_points, current_points)
                max_distances = np.min(distances, axis=1)

                sort_dist_ind = sorted(range(len(max_distances)), key=lambda k: max_distances[k])

                md = int(np.ceil(len(max_distances) / 2.0))
                uq = int(np.ceil(len(max_distances) / 4.0))
                lw = int(np.ceil(len(max_distances) / 8.0))

                index_1 = n_restarts / 2
                index_2 = n_restarts - index_1
                index_1 = np.random.choice(range(uq, md), index_1, replace=False)
                index_2 = np.random.choice(range(lw, uq), index_2, replace=False)

                index_1 = [sort_dist_ind[t] for t in index_1]
                index_2 = [sort_dist_ind[t] for t in index_2]
                index = index_1 + index_2

                start_points = []
                for i in index:
                    start_points.append(new_points[i])
                start_points = np.array(start_points)

            if n_restarts > 0:
                start = np.array(start_points)
                if n_best_restarts > 0 and n_best_restarts < n_restarts:
                    candidate_points = []
                    for i in xrange(n_restarts):
                        candidate_points.append(start[i, :])
                    candidate_points = np.array(candidate_points)

                    output = self.evaluate_mc_bayesian_candidate_points_no_restarts(
                        candidate_points, n_parameters, default_n_samples, default_restarts_mc,
                        n_threads=0, compute_max_mean=True,
                        compute_gradient=False, method_opt=method_opt_mc, **opt_params_mc)

                    evaluations = output['evaluations']

                    values = values_ei + list(evaluations)
                    values_index = sorted(range(len(values)), key=lambda k: values[k])
                    values_index = values_index[-n_best_restarts:]
                    start_ = []
                    for j in values_index:
                        if j < len(values_ei):
                            start_.append(points_st_ei[j])
                        else:
                            start_.append(start[j - len(values_ei), :])
                    start = np.array(start_)
                if start_ei:
                    start = np.concatenate((start, st_ei), axis=0)
            else:
                start = st_ei

            n_restarts = start.shape[0]
        else:
            n_restarts = 1

        bounds = [tuple(bound) for bound in self.bounds_opt]
        opt_method = None
        compute_value_function = False
        if n_samples_parameters == 0 and not monte_carlo:
            #TODO: CHECK THIS
            optimization = Optimization(
                LBFGS_NAME,
                wrapper_objective_voi,
                bounds,
                wrapper_gradient_voi,
                minimize=False, **{'maxiter': 10})

            if n_restarts > int( mp.cpu_count() / 2):
                args = (False, None, parallel, 0, optimization, self, monte_carlo, n_samples,
                        n_restarts_mc, n_best_restarts_mc, opt_params_mc, n_threads,
                        n_samples_parameters, method_opt_mc)
            else:
                args = (False, None, False, 0, optimization, self, monte_carlo, n_samples,
                        n_restarts_mc, n_best_restarts_mc,
                        opt_params_mc, 0, n_samples_parameters, method_opt_mc)
            opt_method = wrapper_optimize
            kwargs = {}

            point_dict = {}
            for j in xrange(n_restarts):
                point_dict[j] = start[j, :]
        else:

            args_ = (self, monte_carlo, default_n_samples, default_restarts_mc, n_best_restarts_mc,
                     opt_params_mc, n_threads, n_parameters, method_opt_mc)
            #TODO CHANGE wrapper_objective_voi, wrapper_grad_voi_sgd TO NO SOLVE MAX_a_{n+1} in
            #TODO: parallel for the several starting points

            optimization = Optimization(
                SGD_NAME,
                wrapper_objective_voi,
                bounds,
                wrapper_grad_voi_sgd,
                minimize=False,
                full_gradient=wrapper_gradient_voi,
                args=args_, debug=False, simplex_domain=self.bq.simplex_domain,
                **{'maxepoch': maxepoch}
            )
            #TODO: THINK ABOUT N_THREADS. Do we want to run it in parallel?
            N = max(n_samples * n_samples_parameters, n_samples_parameters, n_samples)
            bayesian = True
            if n_samples_parameters == 0:
                bayesian = False

            args = (False, None, parallel, 0, optimization, N, self, monte_carlo, bayesian,
                    n_restarts_mc, n_best_restarts_mc, n_threads, False, method_opt_mc)

            kwargs = opt_params_mc
            opt_method = wrapper_sgd

            compute_value_function = True

            random_seeds = np.random.randint(0, 4294967295, n_restarts)
            point_dict = {}
            for j in xrange(n_restarts):
                point_dict[j] = [start[j, :], random_seeds[j]]

        optimal_solutions = Parallel.run_function_different_arguments_parallel(
            opt_method, point_dict, *args, **kwargs)

        if compute_value_function:
            candidate_points = []
            for j in xrange(n_restarts):
                try:
                    point = optimal_solutions.get(j)['solution']
                except Exception as e:
                    logger.info("Error optimizing VOI", *self.args_handler)
                    logger.info("Posterior parameters are: ", *self.args_handler)
                    logger.info(self.bq.gp.samples_parameters, *self.args_handler)
                    logger.info("Point is: ", *self.args_handler)
                    logger.info(point_dict[j], *self.args_handler)
                    sys.exit(1)
                candidate_points.append(point)
            candidate_points = np.array(candidate_points)

            logger.info("candidate solutions are: ")
            logger.info(candidate_points)

            output = self.evaluate_mc_bayesian_candidate_points_no_restarts(
                candidate_points, n_parameters, default_n_samples, default_restarts_mc,
                n_threads=0, compute_max_mean=True, compute_gradient=True,
                method_opt=method_opt_mc, **opt_params_mc)

            evaluations = output['evaluations']
            gradients = output['gradient']

            for j in xrange(n_restarts):
                optimal_solutions[j]['optimal_value'] = evaluations[j]
                optimal_solutions[j]['gradient'] = gradients[j, :]

        maximum_values = []
        for j in xrange(n_restarts):
            maximum_values.append(optimal_solutions.get(j)['optimal_value'])

        ind_max = np.argmax(maximum_values)

        # if self.bq.task_continue:
        #     solution = optimal_solutions.get(ind_max)['solution']
        #     for i in range(len(self.bq.gp.type_bounds)):
        #         if self.bq.gp.type_bounds[i] == 1:
        #             value = solution[i]
        #             bounds = self.bq.gp.bounds[i]
        #             new_value = bounds[np.argmin(np.abs(np.array(bounds) - value))]
        #             solution[i] = new_value
        #     optimal_solutions[ind_max]['solution'] = solution

        logger.info("Results of the optimization of the SBO: ", *self.args_handler)
        logger.info(optimal_solutions.get(ind_max), *self.args_handler)

        vect_gradient = optimal_solutions.get(ind_max)['gradient']
        if vect_gradient == 'unavailable' or np.any(np.isnan(vect_gradient)):
            point = optimal_solutions.get(ind_max)['solution']

            if not self.bq.task_continue and self.bq.separate_tasks:
                task = point[-1]
                point = point[0: len(point) - 1]

            norm_point = np.sqrt(np.sum(point ** 2))
            perturbation = norm_point * 1e-6
            parameters_uniform = []
            for i in range(len(bounds)):
                bound = bounds[i]
                dist = point[i] - bound[0]
                lb = min(perturbation, dist)
                dist = bound[1] - point[i]
                ub = min(perturbation, dist)
                parameters_uniform.append([-lb, ub])

            perturbation = []
            for i in range(len(point)):
                lb = parameters_uniform[i][0]
                ub = parameters_uniform[i][1]
                perturbation.append(np.random.uniform(lb, ub))
            perturbation = np.array(perturbation)
            point = point + perturbation

            if not self.bq.task_continue and self.bq.separate_tasks:
                point = np.concatenate((point, [task]))

            optimal_solutions.get(ind_max)['solution'] = point
            optimal_solutions.get(ind_max)['gradient'] = 'unavailable'

        self.optimization_results.append(optimal_solutions.get(ind_max))

        return optimal_solutions.get(ind_max)

    @staticmethod
    def hvoi (b,c,keep):
        M=len(keep)
        if M>1:
            c=c[keep+1]
            c2=-np.abs(c[0:M-1])
            tmp=norm.pdf(c2)+c2*norm.cdf(c2)
            return np.sum(np.diff(b[keep])*tmp)
        else:
            return 0

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.bq.clean_cache()
        self.samples = None
        self.optimal_samples = {}
        self.starting_points_sbo = None
        self.mc_bayesian = {}
        self.bq.optimal_solutions = {}

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         monte_carlo=False, n_samples_parameters=0):
        """
        Write the results of the optimization.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param monte_carlo: (boolean)
        :param n_samples_parameters: int
        """
        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        f_name = self._filename(model_type=model_type,
                                problem_name=problem_name,
                                type_kernel=kernel_name,
                                training_name=training_name,
                                n_training=n_training,
                                random_seed=random_seed,
                                monte_carlo=monte_carlo,
                                n_samples_parameters=n_samples_parameters)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write(self.optimization_results, debug_path)

    def generate_evaluations(self, problem_name, model_type, training_name, n_training,
                             random_seed, iteration, n_points_by_dimension=None, monte_carlo=False,
                             n_samples=1, n_restarts_mc=1):
        """
        Generates evaluations of SBO, and write them in the debug directory.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param iteration: (int)
        :param n_points_by_dimension: [int] Number of points by dimension
        :param monte_carlo: (boolean) If True, estimates the objective function and gradient by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts_mc: (int) Number of restarts to optimize a_{n+1} given a sample.

        """

        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]


        f_name = self._filename_points_voi_evaluations(
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        vectors = JSONFile.read(debug_path)


        if vectors is None:
            bounds = self.bq.gp.bounds
            n_points = n_points_by_dimension
            if n_points is None:
                n_points = (bounds[0][1] - bounds[0][0]) * 10

            bounds_x = [bounds[i] for i in xrange(len(bounds)) if i in self.bq.x_domain]
            n_points_x = [n_points[i] for i in xrange(len(n_points)) if i in self.bq.x_domain]

            points = []
            for bound, number_points in zip(bounds_x, n_points_x):
                points.append(np.linspace(bound[0], bound[1], number_points))

            vectors = []
            for point in itertools.product(*points):
                vectors.append(point)

            JSONFile.write(vectors, debug_path)


        # TODO: extend to the case where w can be continuous
        n = len(vectors)
        points = deepcopy(vectors)
        vectors = np.array(vectors)

        # point_dict = {}
        # for i in xrange(n):
        #     point_dict[i] = vectors[i:i+1, :]

        values = {}
        if self.bq.tasks:
            for task in xrange(self.bq.n_tasks):
                if not monte_carlo:
                    values[task] = wrapper_evaluate_sbo(vectors, task, self)
                else:
                    values[task] = wrapper_evaluate_sbo_mc(
                        vectors, task, self, n_samples=n_samples, n_restarts=n_restarts_mc)


        f_name = self._filename_voi_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed,
                                                monte_carlo=monte_carlo)

        debug_path = path.join(debug_dir, f_name)


        JSONFile.write({'points': points, 'evaluations': values}, debug_path)

        return values

