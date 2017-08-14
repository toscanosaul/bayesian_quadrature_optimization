from __future__ import absolute_import

import numpy as np

import itertools

from os import path
import os

from copy import deepcopy

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import (
    UNIFORM_FINITE,
    TASKS,
    QUADRATURES,
    POSTERIOR_MEAN,
    TASKS_KERNEL_NAME,
    LBFGS_NAME,
    DEBUGGING_DIR,
    B_NEW,
    BAYESIAN_QUADRATURE,
    SBO_METHOD,
)
from stratified_bayesian_optimization.lib.la_functions import (
    cho_solve,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.lib.expectations import (
    uniform_finite,
    gradient_uniform_finite,
    gradient_uniform_finite_resp_candidate,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    wrapper_evaluate_quadrature_cross_cov,
    wrapper_compute_vector_b,
    wrapper_objective_posterior_mean_bq,
    wrapper_grad_posterior_mean_bq,
    wrapper_optimize,
)

logger = SBOLog(__name__)


class BayesianQuadrature(object):
    _filename = 'opt_post_mean_gp_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}_{method}.json'.format

    _filename_mu_evaluations = '{iteration}_post_mean_gp_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format


    _filename_points_mu_evaluations = 'points_for_post_mean_gp_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format

    _expectations_map = {
        UNIFORM_FINITE: {
            'expectation': uniform_finite,
            'grad_expectation': gradient_uniform_finite,
            'parameter': TASKS,
            'grad_expectation_candidate': gradient_uniform_finite_resp_candidate,
        },
    }

    def __init__(self, gp_model, x_domain, distribution, parameters_distribution=None,
                 model_only_x=False):
        """

        :param gp_model: gp_fitting_gaussian instance
        :param x_domain: [int], indices of the x domain
        :param distribution: (str), it must be in the list of distributions:
            [UNIFORM_FINITE]
        :param parameters_distribution: (dict) dictionary with parameters of the distribution.
            -UNIFORM_FINITE: dict{TASKS: int}
        :param model_only_x (boolean) If True, we keep only the type bounds and bounds of x. So,
            we can use BQ with other methods like EI.
        """
        self.gp = gp_model
        self.name_model = BAYESIAN_QUADRATURE
        if parameters_distribution == {}:
            parameters_distribution = None

        if parameters_distribution is None and distribution == UNIFORM_FINITE:
            for name in self.gp.kernel.names:
                if name == TASKS_KERNEL_NAME:
                    n = self.gp.kernel.kernels[name].n_tasks
                    break
            parameters_distribution = {TASKS: n}

        self.parameters_distribution = parameters_distribution
        self.dimension_domain = self.gp.dimension_domain
        self.x_domain = x_domain
        # Indices of the w_domain
        self.w_domain = [i for i in range(self.gp.dimension_domain) if i not in x_domain]
        self.expectation = self._expectations_map[distribution]
        self.distribution = distribution

        self.tasks = False
        if self.distribution == UNIFORM_FINITE:
            self.tasks = True
            self.n_tasks = self.parameters_distribution.get(TASKS)

        self.arguments_expectation = {}

        if self.expectation['parameter'] == TASKS:
            n_tasks = self.parameters_distribution.get(TASKS)
            self.arguments_expectation['domain_random'] = np.arange(n_tasks).reshape((n_tasks, 1))

        self.cache_quadratures = {}
        self.cache_posterior_mean = {}
        self.cache_quadrature_with_candidate = {}
        self.optimal_solutions = {} # The optimal solutions are written here

        # Cached data for the MC estimation of the SBO.
        self.cache_sample = {}
        self.max_mean = {}


        self.separate_tasks = False

        if self.gp.type_bounds != [] and self.gp.type_bounds[-1] == 1 and not model_only_x:
            self.separate_tasks = True
        self.model_only_x = model_only_x

        if model_only_x or self.separate_tasks:
            self.type_bounds = [
                self.gp.type_bounds[i] for i in xrange(len(self.gp.type_bounds))
                if i in self.x_domain]
        else:
            self.type_bounds = deepcopy(self.gp.type_bounds)


        self.tasks = []
        if self.gp.bounds != [] and self.separate_tasks:
            self.tasks = self.gp.bounds[-1]
        if model_only_x or self.separate_tasks:
            self.bounds = [
                self.gp.bounds[i] for i in xrange(len(self.gp.bounds)) if i in self.x_domain]
        else:
            self.bounds = deepcopy(self.gp.bounds)

        self.type_kernel = self.gp.type_kernel

        # Used to compute the best solution for EI.
        self.best_solution = None


    def _get_cached_data(self, index, name):
        """
        :param index: tuple. (parameters_kernel, )
        :param name: (str) QUADRATURES or POSTERIOR_MEAN or B_NEW

        :return: cached data if it's cached, otherwise False
        """

        if name == QUADRATURES:
            if index in self.cache_quadratures:
                return self.cache_quadratures[index]
        if name == POSTERIOR_MEAN:
            if index in self.cache_posterior_mean:
                return self.cache_posterior_mean[index]
        if name == B_NEW:
            if index in self.cache_quadrature_with_candidate:
                return self.cache_quadrature_with_candidate[index]
        return None

    def _updated_cached_data(self, index, value, name, thread=False):
        """

        :param index: tuple. (parameters_kernel, )
        :param value: value to be cached
        :param name: (str) QUADRATURES or POSTERIOR_MEAN or B_NEW
        :param thread: (boolean) True if memory is shared between threads.

        """

        if name == QUADRATURES:
            if not thread:
                self.cache_quadratures = {}
            self.cache_quadratures[index] = value
        if name == POSTERIOR_MEAN:
            if not thread:
                self.cache_posterior_mean = {}
            self.cache_posterior_mean[index] = value
        if name == B_NEW:
            if not thread:
                self.cache_quadrature_with_candidate = {}
            self.cache_quadrature_with_candidate[index] = value

    def evaluate_quadrate_cov(self, point, parameters_kernel):
        """
        Evaluate the quadrature cov, i.e.
            Expectation(cov((point, W))) respect to W.

        :param point: np.array(1xk)
        :param parameters_kernel: np.array(l)
        :return: np.array(m)
        """
        f = lambda x: self.gp.evaluate_cov(x, parameters_kernel)

        parameters = {
            'f': f,
            'point': point,
            'index_points': self.x_domain,
            'index_random': self.w_domain,
            'double': True,
        }

        parameters.update(self.arguments_expectation)

        return self.expectation['expectation'](**parameters)

    def evaluate_quadrature_cross_cov(self, point, points_2, parameters_kernel):
        """
        Evaluate the quadrature cross cov respect to point, i.e.
            Expectation(cov((x_i,w_i), (x'_j,w'_j))) respect to w_i, where point = (x_i), and
            points_2 = (x'_j, w'_j).
        This is [B(x, j)] in the SBO paper.

        :param point: np.array(txk)
        :param points_2: np.array(mxk')
        :param parameters_kernel: np.array(l)
        :return: np.array(txm)
        """

        f = lambda x: self.gp.evaluate_cross_cov(x, points_2, parameters_kernel)

        parameters = {
            'f': f,
            'point': point,
            'index_points': self.x_domain,
            'index_random': self.w_domain,
        }

        parameters.update(self.arguments_expectation)

        B = self.expectation['expectation'](**parameters)

        return B

    def evaluate_grad_quadrature_cross_cov(self, point, points_2, parameters_kernel):
        """
        Evaluate the gradient respect to the point of the quadrature cross cov i.e.
            gradient(Expectation(cov((x_i,w_i), (x'_j,w'_j)))), where point = (x_i), and
            points_2 = (x'_j, w'_j).
        This is gradient[B(x, j)] respect to x=point in the SBO paper.

        :param point: np.array(1xk)
        :param points_2: np.array(mxk')
        :param parameters_kernel: np.array(l)
        :return: np.array(kxm)
        """

        parameters = {
            'f': self.gp.evaluate_grad_cross_cov_respect_point,
            'point': point,
            'points_2': points_2,
            'index_points': self.x_domain,
            'index_random': self.w_domain,
            'parameters_kernel': parameters_kernel,
        }

        parameters.update(self.arguments_expectation)

        gradient = self.expectation['grad_expectation'](**parameters)

        return gradient

    def evaluate_grad_quadrature_cross_cov_resp_candidate(self, candidate_point, points,
                                                          parameters_kernel):
        """
        Evaluate the gradient respect to the candidate_point of the quadrature cross cov i.e.
            gradient(Expectation(cov((x_i,w_i), candidate_point))), where point = (x_i) is in
            points, and candidate_point = (x, w).
        This is gradient[B(x, n+1)] respect to n+1 in the SBO paper.

        :param candidate_point: np.array(1xk)
        :param points: np.array(mxk')
        :param parameters_kernel: np.array(l)
        :return: np.array(kxm)
        """

        parameters = {
            'f': self.gp.evaluate_grad_cross_cov_respect_point,
            'candidate_point': candidate_point,
            'points': points,
            'index_points': self.x_domain,
            'index_random': self.w_domain,
            'parameters_kernel': parameters_kernel,
        }

        parameters.update(self.arguments_expectation)

        gradient = self.expectation['grad_expectation_candidate'](**parameters)

        return gradient

    def compute_posterior_parameters(self, points, var_noise=None, mean=None,
                                     parameters_kernel=None, historical_points=None,
                                     historical_evaluations=None, only_mean=False, cache=True,
                                     parallel=False):
        """
        Compute posterior mean and covariance of the GP on G(x) = E[F(x, w)] evaluated at each point
        of points.

        :param points: np.array(txk) More than one point only if only_mean is True!
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param historical_points: np.array(nxm)
        :param historical_evaluations: np.array(n)
        :param cache: (boolean) get cached data only if cache is True
        :param only_mean: (boolean) computes only the mean if it's True.
        :param parallel: (boolean) computes the vector B(x, i) in parallel for every point in
            points

        :return: {
            'mean': np.array(t),
            'cov': float,
        }
        """

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        if historical_points is None:
            historical_points = self.gp.data['points']

        if historical_evaluations is None:
            historical_evaluations = self.gp.data['evaluations']

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, historical_points=historical_points,
            historical_evaluations=historical_evaluations, cache=cache)

        solve = chol_solve['solve']
        chol = chol_solve['chol']


        n = points.shape[0]
        m = historical_points.shape[0]

        compute_vec_covs = False
        if cache and points.shape[0] == 1:
            vec_covs = self._get_cached_data(
                (tuple(parameters_kernel), tuple(points[0, :])), QUADRATURES)
        else:
            vec_covs = None

        if vec_covs is None:
            compute_vec_covs = True
            vec_covs = np.zeros((n, m))

        if compute_vec_covs:
            computations = self.compute_vectors_b(points, None, historical_points,
                                                  parameters_kernel, compute_vec_covs,
                                                  False, parallel)

            vec_covs = computations['vec_covs']

        if cache and compute_vec_covs and points.shape[0] == 1:
            self._updated_cached_data(
                (tuple(parameters_kernel), tuple(points[0, :])), vec_covs, QUADRATURES)

        mu_n = mean + np.dot(vec_covs, solve)

        if only_mean:
            return {
            'mean': mu_n,
            'cov': None,
        }

        solve_2 = cho_solve(chol, vec_covs.transpose())

        cov_n = self.evaluate_quadrate_cov(points, parameters_kernel) - np.dot(vec_covs, solve_2)

        return {
            'mean': mu_n,
            'cov': cov_n[0, 0],
        }

    def gradient_posterior_mean(self, point, var_noise=None, mean=None, parameters_kernel=None,
                                historical_points=None, historical_evaluations=None, cache=True):
        """
        Compute gradient of the posterior mean of the GP on G(x) = E[F(x, w)].

        :param point: np.array(1xk)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param historical_points: np.array(nxm)
        :param historical_evaluations: np.array(n)
        :param cache: (boolean) get cached data only if cache is True

        :return: np.array(k)
        """

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        if historical_points is None:
            historical_points = self.gp.data['points']

        if historical_evaluations is None:
            historical_evaluations = self.gp.data['evaluations']

        gradient = self.evaluate_grad_quadrature_cross_cov(point, historical_points,
                                                           parameters_kernel)

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, historical_points=historical_points,
            historical_evaluations=historical_evaluations, cache=cache)

        solve = chol_solve['solve']

        return np.dot(gradient, solve)

    def gradient_posterior_parameters(self, point, var_noise=None, mean=None,
                                      parameters_kernel=None, cache=True):
        """
        Computes the gradient of the posterior parameters of the GP on g(x).

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: boolean

        :return: {'mean': np.array(n), 'cov': np.array(n)}
        """

        # We assume that cov(x, x) is constant respect to x (it's a radial kernel)
        # TODO: WE CAN CACHE VEC_COV

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        gradient = self.evaluate_grad_quadrature_cross_cov(point, self.gp.data['points'],
                                                           parameters_kernel)

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, cache=True)

        compute_vec_covs = False
        if cache:
            vec_covs = self._get_cached_data(
                (tuple(parameters_kernel), tuple(point[0, :])), QUADRATURES)
        else:
            vec_covs = None

        if vec_covs is None:
            compute_vec_covs = True

        if compute_vec_covs:
            computations = self.compute_vectors_b(point, None, self.gp.data['points'],
                                                  parameters_kernel, compute_vec_covs,
                                                  False, True)

            if compute_vec_covs:
                vec_covs = computations['vec_covs']

        if cache and compute_vec_covs:
            self._updated_cached_data(
                (tuple(parameters_kernel), tuple(point[0, :])), vec_covs, QUADRATURES)

        solve = chol_solve['solve']
        chol = chol_solve['chol']

        grad_mu = np.dot(gradient, solve)
        solve_3 = cho_solve(chol, gradient.transpose())
        grad_cov = - 2.0 * np.dot(vec_covs, solve_3)

        return {'mean': grad_mu, 'cov': grad_cov}

    def objective_posterior_mean(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Computes the posterior mean evaluated on point.

        :param point: np.array(k)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: float
        """

        point = point.reshape((1, len(point)))

        return self.compute_posterior_parameters(point, var_noise=var_noise, mean=mean,
                                                 parameters_kernel=parameters_kernel,
                                                 only_mean=True)['mean']

    def grad_posterior_mean(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Computes the gradient of the posterior mean evaluated on point.

        :param point: np.array(k)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: np.array(k)
        """

        point = point.reshape((1, len(point)))
        return self.gradient_posterior_mean(point, var_noise=var_noise, mean=mean,
                                            parameters_kernel=parameters_kernel)

    def optimize_posterior_mean(self, start=None, random_seed=None, minimize=False, n_restarts=1000,
                                parallel=True, n_treads=0, var_noise=None, mean=None,
                                parameters_kernel=None):
        """
        Optimize the posterior mean.

        :param start: np.array(n)
        :param random_seed: float
        :param minimize: boolean
        :param n_restarts: int
        :param parallel: (boolean)
        :param n_treads: (int)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: dictionary with the results of the optimization
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        bounds_x = [self.gp.bounds[i] for i in xrange(len(self.gp.bounds)) if i in
                    self.x_domain]

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        index_cache = (var_noise, mean, tuple(parameters_kernel))

        if start is None:
            start_points = DomainService.get_points_domain(n_restarts + 1, bounds_x,
                                                        type_bounds=len(self.x_domain) * [0])
            if index_cache in self.optimal_solutions and \
                            len(self.optimal_solutions[index_cache]) > 0:
                start = self.optimal_solutions[index_cache][-1]['solution']

                start = [start] + start_points[0: -1]
                start = np.array(start)
            else:
                start = np.array(start_points)

        bounds = [tuple(bound) for bound in bounds_x]

        objective_function = wrapper_objective_posterior_mean_bq
        grad_function = wrapper_grad_posterior_mean_bq

        optimization = Optimization(
            LBFGS_NAME,
            objective_function,
            bounds,
            grad_function,
            minimize=minimize)

        point_dict = {}
        for j in xrange(n_restarts + 1):
            point_dict[j] = start[j, :]

        args = (False, None, parallel, n_treads, optimization, self, var_noise, mean,
                parameters_kernel)

        optimal_solutions = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, point_dict, *args)

        maximum_values = []
        for j in xrange(n_restarts + 1):
            maximum_values.append(optimal_solutions.get(j)['optimal_value'])

        max_ = np.max(maximum_values)
        ind_max = np.argmax(maximum_values)

        logger.info("Results of the optimization of the posterior mean: ")
        logger.info(optimal_solutions.get(ind_max))

        if index_cache not in self.optimal_solutions:
            self.optimal_solutions[index_cache] = []

        self.optimal_solutions[index_cache].append(optimal_solutions.get(ind_max))
        self.max_mean[index_cache] = max_
        return optimal_solutions.get(ind_max)

    def compute_vectors_b(self, points, candidate_points, historical_points, parameters_kernel,
                          compute_vec_covs, compute_b_new, parallel, n_threads=0):
        """
        Compute B(x, i) for ever x in points, and B(candidate_point, i) for each i.

        :param points: np.array(nxk)
        :param candidate_points: np.array(kxm), (new_x, new_w)
        :param parameters_kernel: np.array(l)
        :param historical_points: np.array(txm)
        :param parameters_kernel: np.array(l)
        :param compute_vec_covs: boolean
        :param compute_b_new: boolean
        :param parallel: boolean
        :param n_threads: (int)

        :return: {
            'b_new': np.array(nxk),
            'vec_covs': np.array(nxt),
        }
        """


        n = points.shape[0]
        m = historical_points.shape[0]


        vec_covs = None
        b_new = None

        if compute_vec_covs:
            vec_covs = np.zeros((n, m))

        if compute_b_new:
            n_candidate_points = candidate_points.shape[0]
            b_new = np.zeros((n, n_candidate_points))

        if parallel:
            point_dict = {}
            for i in xrange(n):
                point_dict[i] = points[i:i + 1, :]

            args = (False, None, True, n_threads, compute_vec_covs, compute_b_new,
                    historical_points, parameters_kernel, candidate_points, self,)

            b_vectors = Parallel.run_function_different_arguments_parallel(
                wrapper_compute_vector_b, point_dict, *args)

            for i in xrange(n):
                if b_vectors.get(i) is None:
                    logger.info("Error in computing b vectors at point %d" % i)
                    continue
                if compute_vec_covs:
                    vec_covs[i, :] = b_vectors[i]['vec_covs']
                if compute_b_new:
                    b_new[i, :] = b_vectors[i]['b_new']
        else:
            for i in xrange(n):
                if compute_vec_covs:
                    vec_covs[i, :] = self.evaluate_quadrature_cross_cov(
                        points[i:i + 1, :], historical_points, parameters_kernel)
                if compute_b_new:
                    b_new[i, :] = self.evaluate_quadrature_cross_cov(
                        points[i:i + 1, :], candidate_points, parameters_kernel)

        return {'b_new': b_new, 'vec_covs': vec_covs}


    def compute_posterior_parameters_kg_many_cp(self, points, candidate_points, cache=True,
                                                parallel=True):
        """
        Compute posterior parameters of the GP after integrating out the random parameters needed
        to compute the knowledge gradient (vectors "a" and "b" in the SBO paper).

        :param points: np.array(nxk)
        :param candidate_points: np.array(rxm), (new_x, new_w)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param parallel: (boolean) compute B(x, i) in parallel for all x in points

        :return: {
            'a': np.array(n),
            'b': np.array(nxr)
        }
        """

        var_noise = self.gp.var_noise.value[0]
        parameters_kernel = self.gp.kernel.hypers_values_as_array
        mean = self.gp.mean.value[0]

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, cache=cache)
        chol = chol_solve['chol']
        solve = chol_solve['solve']

        n = points.shape[0]
        m = self.gp.data['points'].shape[0]
        n_new_points = candidate_points.shape[0]

        b_new = None
        compute_b_new = False
        if b_new is None:
            compute_b_new = True
            b_new = np.zeros((n, n_new_points))

        compute_vec_covs = False
        if cache:
            vec_covs = self._get_cached_data((tuple(parameters_kernel, )), QUADRATURES)
        else:
            vec_covs = None

        if vec_covs is None:
            compute_vec_covs = True
            vec_covs = np.zeros((n, m))

        if compute_vec_covs or compute_b_new:
            computations = self.compute_vectors_b(points, candidate_points, self.gp.data['points'],
                                                  parameters_kernel, compute_vec_covs,
                                                  compute_b_new, parallel)

            if compute_vec_covs:
                vec_covs = computations['vec_covs']

            if compute_b_new:
                b_new = computations['b_new']

        if cache:
            if compute_vec_covs:
                self._updated_cached_data((tuple(parameters_kernel), ), vec_covs, QUADRATURES)

        if cache:
            mu_n = self._get_cached_data((tuple(parameters_kernel),), POSTERIOR_MEAN)
        else:
            mu_n = None

        if mu_n is None:
            mu_n = mean + np.dot(vec_covs, solve)
            if cache:
                self._updated_cached_data((tuple(parameters_kernel),), mu_n, POSTERIOR_MEAN)

        # TODO: CACHE SO WE DON'T COMPUTE MU_N ALL THE TIME
        cross_cov = self.gp.evaluate_cross_cov(self.gp.data['points'], candidate_points,
                                                parameters_kernel)

        solve_2 = cho_solve(chol, cross_cov)

        numerator = b_new - np.dot(vec_covs, solve_2)

        new_cross_cov = np.diag(self.gp.evaluate_cross_cov(candidate_points, candidate_points,
                                                   parameters_kernel))

        denominator = new_cross_cov - np.einsum('ij,ij->j', cross_cov, solve_2)
        denominator = np.clip(denominator, 0, None)
        denominator = np.sqrt(denominator)

        b_value = numerator / denominator[None, :]

        return {
            'a': mu_n,
            'b': b_value,
        }

    def get_parameters_for_samples(self, cache, candidate_point, parameters_kernel,
                                           var_noise, mean):
        """
        Computes additional parameters needed for sample of SBO.

        :param cache: (boolean)
        :param candidate_point: np.array(1xm)
        :param parameters_kernel: np.array(l)
        :param var_noise: float
        :param mean: float
        :return: {
            'gamma': cov(historical_points, candidate_point),
            'solve_2': cov(historical_points)^-1 * gamma,
            'denominator': np.sqrt(cov(candidate_point) - np.diag(gamma, solve_2)),
            'chol': chol(cov(historical_points)),
            'solve': cov(historical_points)^-1 * (y_historical),
        }
        """


        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, cache=cache)
        chol = chol_solve['chol']
        solve = chol_solve['solve']

        index_cache = (tuple(candidate_point[0, :]), tuple(parameters_kernel))
        if cache and index_cache in self.cache_sample:
            solve_2 = self.cache_sample[index_cache]['solve_2']
            denominator = self.cache_sample[index_cache]['denominator']
            cross_cov = self.cache_sample[index_cache]['gamma']
        else:

            cross_cov = self.gp.evaluate_cross_cov(self.gp.data['points'], candidate_point,
                                                   parameters_kernel)  # cache this

            solve_2 = cho_solve(chol, cross_cov)

            new_cross_cov = np.diag(self.gp.evaluate_cross_cov(candidate_point, candidate_point,
                                                       parameters_kernel))
            denominator = new_cross_cov - np.einsum('ij,ij->j', cross_cov, solve_2)
            denominator = np.clip(denominator, 0, None)
            denominator = np.sqrt(denominator)
            if cache:
                self.cache_sample = {}
                self.cache_sample[index_cache] = {}
                self.cache_sample[index_cache]['denominator'] = denominator
                self.cache_sample[index_cache]['solve_2'] = solve_2
                self.cache_sample[index_cache]['gamma'] = cross_cov

        return {
            'gamma': cross_cov,
            'solve_2': solve_2,
            'denominator': denominator,
            'chol': chol,
            'solve': solve,
            'parameters_kernel': parameters_kernel,
            'var_noise': var_noise,
            'mean': mean,
        }


    def compute_parameters_for_sample(
            self, point, candidate_point, var_noise=None, mean=None,
            parameters_kernel=None, cache=True, n_threads=0):
        """
        Compute posterior parameters of a_n+1(point) given the candidate_point. Caching is different
        than in the other functions.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param n_threads: (int) Threads are used if n_threads > 0
        :return: {'a': float, 'b': float}
        """

        additional_parameters = self.get_parameters_for_samples(
            cache, candidate_point, parameters_kernel, var_noise, mean)


        chol = additional_parameters.get('chol')
        solve = additional_parameters.get('solve')
        parameters_kernel = additional_parameters.get('parameters_kernel')
        mean = additional_parameters.get('mean')
        var_noise = additional_parameters.get('var_noise')

        vec_covs, b_new = self.get_vec_covs(cache, point, parameters_kernel, candidate_point,
                                            False, monte_carlo=True, n_threads=n_threads)

        mu_n = mean + np.dot(vec_covs, solve)

        solve_2 = additional_parameters.get('solve_2')
        denominator = additional_parameters['denominator']

        numerator = b_new - np.dot(vec_covs, solve_2)
        b_value = numerator / denominator[None, :]

        return {'a': mu_n, 'b': b_value}

    def compute_gradient_parameters_for_sample(
            self, point, candidate_point, var_noise=None, mean=None,
            parameters_kernel=None, cache=True):
        """
        Compute the gradient of the posterior parameters of a_n+1(point) respect to point given the
        candidate_point.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :return: {'a': np.array(n), 'b': np.array(n)}
        """

        additional_parameters = self.get_parameters_for_samples(
            cache, candidate_point, parameters_kernel, var_noise, mean)

        chol = additional_parameters.get('chol')
        solve = additional_parameters.get('solve')
        parameters_kernel = additional_parameters.get('parameters_kernel')
        mean = additional_parameters.get('mean')
        var_noise = additional_parameters.get('var_noise')


        historical_points = self.gp.data['points']
        historical_evaluations = self.gp.data['evaluations']


        gradient = self.evaluate_grad_quadrature_cross_cov(point, historical_points,
                                                           parameters_kernel)

        gradient_a = np.dot(gradient, solve)

        denominator = additional_parameters.get('denominator')
        solve_2 = additional_parameters.get('solve_2')

        gradient_new = self.evaluate_grad_quadrature_cross_cov(point, candidate_point,
                                                           parameters_kernel)
        gradient_b = (gradient_new - np.dot(gradient, solve_2)) / denominator
        gradient_b = gradient_b[:, 0]
        return {'a': gradient_a, 'b': gradient_b}

    def get_vec_covs(self, cache, points, parameters_kernel, candidate_point, parallel,
                     keep_indexes=None, monte_carlo=False, n_threads=0):
        """
        Get vectors b from cache if possible.

        :param cache: (boolean)
        :param points: np.array(nxk)
        :param parameters_kernel: np.array(l)
        :param candidate_point: np.array(1xm)
        :param parallel: boolean
        :param keep_indexes: [int], indexes of the points saved of the discretization.
            They are used to get the useful elements of the cached data. Monte_carlo is False.
        :param monte_carlo: If True, we cache the data using the indexes to cache the data of the
            monte carlo samples.
        :param n_threads: (int) If n_threads > 0, memory is shared between threads

        :return: (vec_covs, b_new)
        """

        if monte_carlo:
            parallel = False

        n = points.shape[0]
        m = self.gp.data['points'].shape[0]

        if cache:
            if not monte_carlo:
                index_b_new = (tuple(parameters_kernel), tuple(candidate_point[0, :]))
            else:
                index_b_new = (tuple(parameters_kernel), tuple(candidate_point[0, :]),
                               tuple(points[0, :]))

            b_new = self._get_cached_data(index_b_new, B_NEW)
        else:
            b_new = None

        compute_b_new = False
        if b_new is None:
            compute_b_new = True
            b_new = np.zeros((n, 1))
        elif not monte_carlo and keep_indexes is not None:
            b_new = b_new[keep_indexes, :]

        compute_vec_covs = False
        if cache:
            if not monte_carlo:
                index_vec_covs = (tuple(parameters_kernel), )
            else:
                index_vec_covs = (tuple(parameters_kernel), tuple(points[0, :]))

            vec_covs = self._get_cached_data(index_vec_covs, QUADRATURES)
        else:
            vec_covs = None

        if vec_covs is None:
            compute_vec_covs = True
            vec_covs = np.zeros((n, m))
        elif not monte_carlo and keep_indexes is not None:
            vec_covs = vec_covs[keep_indexes, :]

        if compute_vec_covs or compute_b_new:
            computations = self.compute_vectors_b(points, candidate_point, self.gp.data['points'],
                                                  parameters_kernel, compute_vec_covs,
                                                  compute_b_new, parallel, n_threads=n_threads)

            if compute_vec_covs:
                vec_covs = computations['vec_covs']

            if compute_b_new:
                b_new = computations['b_new']

        if cache:
            if n_threads > 0 and monte_carlo:
                thread = True
            else:
                thread = False

            if compute_vec_covs:
                self._updated_cached_data(index_vec_covs, vec_covs, QUADRATURES, thread=thread)

            if compute_b_new:
                self._updated_cached_data(index_b_new, b_new, B_NEW, thread=thread)

        return vec_covs, b_new


    def compute_posterior_parameters_kg(self, points, candidate_point, var_noise=None, mean=None,
                                        parameters_kernel=None, cache=True, parallel=True,
                                        n_threads=0):
        """
        Compute posterior parameters of the GP after integrating out the random parameters needed
        to compute the knowledge gradient (vectors "a" and "b" in the SBO paper).

        :param points: np.array(nxk)
        :param candidate_point: np.array(1xm), (new_x, new_w)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param parallel: (boolean) compute B(x, i) in parallel for all x in points
        :param n_threads: (int)

        :return: {
            'a': np.array(n),
            'b': np.array(n)
        }
        """

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        chol_solve = self.gp._cholesky_solve_vectors_for_posterior(
            var_noise, mean, parameters_kernel, cache=cache)
        chol = chol_solve['chol']
        solve = chol_solve['solve']

        vec_covs, b_new = self.get_vec_covs(cache, points, parameters_kernel, candidate_point,
                                            parallel, n_threads=n_threads)

        if cache:
            mu_n = self._get_cached_data((tuple(parameters_kernel), mean), POSTERIOR_MEAN)
        else:
            mu_n = None

        if mu_n is None:
            mu_n = mean + np.dot(vec_covs, solve)
            if cache:
                self._updated_cached_data((tuple(parameters_kernel), mean), mu_n, POSTERIOR_MEAN)

        # TODO: CACHE SO WE DON'T COMPUTE MU_N ALL THE TIME
        cross_cov = self.gp.evaluate_cross_cov(candidate_point, self.gp.data['points'],
                                                parameters_kernel)

        solve_2 = cho_solve(chol, cross_cov[0, :])
        numerator = b_new[:, 0] - np.dot(vec_covs, solve_2)

        new_cross_cov = self.gp.evaluate_cross_cov(candidate_point, candidate_point,
                                                   parameters_kernel)

        denominator = new_cross_cov - np.dot(cross_cov, solve_2)
        denominator = np.clip(denominator[0, 0], 0, None)

        b_value = numerator / np.sqrt(denominator)

        return {
            'a': mu_n,
            'b': b_value,
        }

    def gradient_vector_b(self, candidate_point, points, var_noise=None, mean=None,
                          parameters_kernel=None, cache=True, keep_indexes=None,
                          parallel=True, monte_carlo=False, n_threads=0):
        """
        Compute the gradient of the vector b(x,candidate_point) for each x in points
        (see SBO paper).

        :param candidate_point: np.array(1xm), (new_x, new_w)
        :param points: np.array(nxk)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :param keep_indexes: [int], indexes of the points saved of the discretization.
            They are used to get the useful elements of the cached data.
        :param parallel: (boolean)
        :param monte_carlo: If True, we cache the data using the indexes to cache the data of the
            monte carlo samples.
        :param n_threads: (int)

        :return: np.array(nxm)
        """
        # We assume that the gradient of cov(x, x) respect to x is equal to zero.
        # We assume that cov(x, y) = cov(y, x).

        additional_parameters = self.get_parameters_for_samples(
            cache, candidate_point, parameters_kernel, var_noise, mean)

        chol = additional_parameters.get('chol')
        solve = additional_parameters.get('solve')
        parameters_kernel = additional_parameters.get('parameters_kernel')
        mean = additional_parameters.get('mean')
        var_noise = additional_parameters.get('var_noise')
        solve_1 = additional_parameters.get('solve_2')[:, 0]
        beta_1 = additional_parameters.get('denominator') ** 2

        historical_points = self.gp.data['points']



        grad_gamma = self.gp.evaluate_grad_cross_cov_respect_point(candidate_point,
                                                                   historical_points,
                                                                   parameters_kernel)
        grad_b_new = self.evaluate_grad_quadrature_cross_cov_resp_candidate(candidate_point,
                                                                            points,
                                                                            parameters_kernel)

        #
        # beta_1 = self.gp.evaluate_cov(candidate_point, parameters_kernel) - \
        #          np.dot(gamma.transpose(), solve_1)

        vec_covs, b_new =  self.get_vec_covs(cache, points, parameters_kernel, candidate_point,
                                             parallel, keep_indexes=keep_indexes,
                                             monte_carlo=monte_carlo, n_threads=n_threads)

        solve_2 = cho_solve(chol, vec_covs.transpose())

        beta_1 =  beta_1 ** (-0.5)

        beta_2 = b_new[:, 0] - np.dot(vec_covs, solve_1)

        beta_3 = grad_b_new - np.dot(grad_gamma.transpose(), solve_2)

        beta_4 = 2.0 * np.dot(grad_gamma.transpose(), solve_1)
        beta_4 = beta_4.reshape((candidate_point.shape[1], 1))

        beta_5 = 0.0

        gradients = beta_1[0] * beta_3 - 0.5 * (beta_1[0] ** 3) * beta_2 * (beta_5 - beta_4)

        return gradients.transpose()

    def sample_new_observations(self, point, n_samples, random_seed=None):
        """
        Sample g(point) = E[f(point,w)] n_samples times.

        :param point: np.array(1xn)
        :param n_samples: int
        :param random_seed: int
        :return: np.array(n_samples)
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        posterior_parameters = self.compute_posterior_parameters(point)
        mean = posterior_parameters['mean']
        var = posterior_parameters['cov']

        samples = np.random.normal(mean, np.sqrt(var), n_samples)

        return samples

    def get_historical_best_solution(self, var_noise=None, mean=None, parameters_kernel=None,
                                     noisy_evaluations=False):
        """
        Computes the best solution so far based on the GP on the objective function
            g(x) = E[f(x, w)]

        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param noisy_evaluations: boolean
        :return: float
        """
        if self.best_solution is not None:
            best = self.best_solution
            return best


        points = self.gp.data['points'][:, self.x_domain]
        evaluations = self.compute_posterior_parameters(
            points, var_noise, mean, parameters_kernel, only_mean=True
        )['mean']

        best = np.max(evaluations)
        self.best_solution = best

        return best

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         method=SBO_METHOD):
        """
        Write information about the different optimizations realized.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param method: (str)
        """

        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        f_name = self._filename(model_type=model_type,
                                problem_name=problem_name,
                                type_kernel=kernel_name,
                                training_name=training_name,
                                n_training=n_training,
                                random_seed=random_seed,
                                method=method)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write(self.optimal_solutions, debug_path)

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.cache_quadratures = {}
        self.cache_posterior_mean = {}
        self.cache_quadrature_with_candidate = {}
        self.gp.clean_cache()
        self.max_mean = {}  # max_{x} a_{n} (x)
        # (a solution for every set of parameters of the model)
        self.best_solution = None

    def generate_evaluations(self, problem_name, model_type, training_name, n_training,
                             random_seed, iteration, n_points_by_dimension=None):
        """
        Generates evaluations of the posterior mean, and write them in the debug directory.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param iteration: (int)
        :param n_points_by_dimension: [int] Number of points by dimension

        """

        # TODO: extend to more than one dimension
        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        f_name = self._filename_points_mu_evaluations(
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        vectors = JSONFile.read(debug_path)

        if vectors is None:
            bounds = self.gp.bounds
            bounds = [bounds[i] for i in xrange(len(bounds)) if i in self.x_domain]

            n_points = n_points_by_dimension
            if n_points is None:
                n_points = (bounds[0][1] - bounds[0][0]) * 10

            points = []
            for bound, number_points in zip(bounds, n_points):
                points.append(np.linspace(bound[0], bound[1], number_points))

            vectors = []
            for point in itertools.product(*points):
                vectors.append(point)

            JSONFile.write(vectors, debug_path)

        vectors = np.array(vectors)

        values = self.compute_posterior_parameters(vectors, only_mean=True, parallel=True)['mean']


        f_name = self._filename_mu_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write({'points': vectors, 'evaluations': values}, debug_path)

