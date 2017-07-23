from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from copy import deepcopy

import itertools

from stratified_bayesian_optimization.lib.constant import (
    UNIFORM_FINITE,
    LBFGS_NAME,
    DEBUGGING_DIR,
)
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
from stratified_bayesian_optimization.lib.util import (
    wrapper_optimization,
    wrapper_objective_voi,
    wrapper_gradient_voi,
    wrapper_evaluate_sbo_by_sample,
    wrapper_opimize,
    wrapper_evaluate_sample,
    wrapper_evaluate_gradient_sample,
)
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.util import wrapper_evaluate_sbo

logger = SBOLog(__name__)


class SBO(object):

    _filename = 'opt_sbo_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}.json'.format

    _filename_voi_evaluations = '{iteration}_sbo_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format

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

        self.bounds_opt = self.bq.gp.bounds
        self.opt_separing_domain = False

        # Bounds or list of number of points of the domain
        self.domain_w = [self.bq.gp.bounds[i] for i in self.bq.w_domain]

        if self.bq.distribution == UNIFORM_FINITE:
            self.opt_separing_domain = True

        self.optimization_results = []
        self.max_mean = None # max_{x} a_{n} (x)

    def evaluate_sample(self, point, candidate_point, sample, var_noise=None, mean=None,
                        parameters_kernel=None, cache=True):
        """
        Evaluate a sample of a_{n+1}(point) given that candidate_point is chosen.

        :param point: np.array(1xn)
        :param candidate_point: np.array(1xm)
        :param sample: (float) a sample from a standard Gaussian r.v.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True
        :return: float
        """

        if len(point.shape) == 1:
            point = point.reshape((1, len(point)))

        vectors = self.bq.compute_parameters_for_sample(
            point, candidate_point, var_noise=var_noise, mean=mean,
            parameters_kernel=parameters_kernel, cache=cache)

        return vectors['a'] + sample * vectors['b']

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
        :return: float
        """

        if len(point.shape) == 1:
            point = point.reshape((1, len(point)))

        gradient_params = self.bq.compute_gradient_parameters_for_sample(
            point, candidate_point, var_noise=var_noise, mean=mean,
            parameters_kernel=parameters_kernel, cache=cache
        )

        grad_a = gradient_params['a']
        grad_b = gradient_params['b']

        return grad_a + sample * grad_b

    def evaluate_sbo_by_sample(self, candidate_point, sample, start=None,
                               var_noise=None, mean=None, parameters_kernel=None, n_restarts=5,
                               parallel=True):
        """
        Optimize a_{n+1}(x)  given the candidate_point and the sample of the Gaussian r.v.

        :param candidate_point: np.array(1xn)
        :param sample: float
        :param start: np.array(m)
        :param n_restarts: (int) Number of restarts of the optimization algorithm.
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param parallel: (boolean) Multi-start optimization in parallel if it's True
        :return: float
        """

        bounds_x = [self.bounds_opt[i] for i in xrange(len(self.bounds_opt)) if i in
                    self.bq.x_domain]

        if start is None:
            start_points =  DomainService.get_points_domain(n_restarts + 1, bounds_x,
                                                        type_bounds=len(bounds_x) * [0])
            if len(self.bq.optimal_solutions) > 0:
                start = self.bq.optimal_solutions[-1]['solution']

                start = [start] + start_points[0 : -1]
                start = np.array(start)
            else:
                start = np.array(start_points)
        else:
            n_restarts = 0

        bounds_x = [tuple(bound) for bound in bounds_x]

        if parallel:
            grad_function = wrapper_evaluate_gradient_sample
            objective_function = wrapper_evaluate_sample
        else:
            objective_function = self.evaluate_sample
            grad_function = self.evaluate_gradient_sample

        optimization = Optimization(
            LBFGS_NAME,
            objective_function,
            bounds_x,
            grad_function,
            minimize=False)

        if parallel:
            solutions = []
            point_dict = {}
            for i in xrange(n_restarts + 1):
                point_dict[i] = start[i, :]

            args = (False, None, True, optimization, self, candidate_point, sample, var_noise, mean,
                    parameters_kernel)

            sol = Parallel.run_function_different_arguments_parallel(
                wrapper_opimize, point_dict, *args)

            for i in xrange(n_restarts + 1):
                if sol.get(i) is None:
                    logger.info("Error in computing optimum of a_{n+1} at one sample at point %d"
                                % i)
                    continue
                solutions.append(sol.get(i)['optimal_value'])
        else:
            solutions = []
            for i in xrange(n_restarts + 1):
                start_ = start[i, :]
                args = (candidate_point, sample, var_noise, mean, parameters_kernel)
                results = optimization.optimize(start_, *args)
                solutions.append(results['optimal_value'])

        return np.max(solutions)

    def evaluate_mc(self, candidate_point,  n_samples, var_noise=None, mean=None,
                    parameters_kernel=None, random_seed=None, parallel=True, n_restarts=10):
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
        :return: {'value': float, 'std': float}
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if self.max_mean is not None:
            max_mean = self.max_mean
        else:
            max_mean = self.bq.optimize_posterior_mean(random_seed)['optimal_value']
            self.max_mean = max_mean

        samples = np.random.normal(0, 1, n_samples)

        max_values = []

        if parallel:
            point_dict = {}
            for i in xrange(n_samples):
                point_dict[i] = samples[i]
            args = (False, None, True, self, candidate_point, None, var_noise, mean,
                    parameters_kernel, n_restarts)

            simulated_values = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sbo_by_sample, point_dict, *args)

            for i in xrange(n_samples):
                if simulated_values.get(i) is None:
                    logger.info("Error in computing simulated value at sample %d" % i)
                    continue
                max_values.append(simulated_values[i])
        else:
            for i in xrange(n_samples):
                max_value = self.evaluate_sbo_by_sample(
                    candidate_point, samples[i], start=None, var_noise=var_noise, mean=mean,
                    parameters_kernel=parameters_kernel, n_restarts=n_restarts)
                max_values.append(max_value)

        self.bq.cache_sample = {}

        return {'value': np.mean(max_values) - max_mean, 'std': np.std(max_values) / n_samples}

    def evaluate(self, point, var_noise=None, mean=None, parameters_kernel=None, cache=True):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True

        :return: float
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache)

        a = vectors['a']
        b = vectors['b']

        if not np.all(np.isfinite(b)):
            return 0.0

        a, b, keep = AffineBreakPointsPrep(a, b)

        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)

        return self.hvoi(b, c, keep1)


    def evaluate_gradient(self, point, var_noise=None, mean=None, parameters_kernel=None,
                          cache=True):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True

        :return: np.array(n)
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache)

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
                                              keep_indexes=keep)

        gradient = np.zeros(point.shape[1])
        for i in xrange(point.shape[1]):
            gradient[i] = np.dot(np.diff(gradients[:, i]), evalC)

        return gradient

    def objective_voi(self, point):
        """
        Evaluates the VOI at point.
        :param point: np.array(n)
        :return: float
        """

        point = point.reshape((1, len(point)))
        return self.evaluate(point)

    def grad_obj_voi(self, point):
        """
        Evaluates the gradient of VOI at point.
        :param point: np.array(n)
        :return: np.array(n)
        """

        point = point.reshape((1, len(point)))
        grad = self.evaluate_gradient(point)

        return grad

    def optimize(self, start=None, random_seed=None, parallel=True):
        """
        Optimize the VOI.
        :param start: np.array(n)
        :param random_seed: int
        :param parallel: (boolean) For several tasks, it's run in paralle if it's True

        :return: dictionary with the results of the optimization.
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if start is None:
            start = DomainService.get_points_domain(1, self.bounds_opt,
                                                    type_bounds=self.bq.gp.type_bounds)

        bounds = [tuple(bound) for bound in self.bounds_opt]

        for i in self.bq.w_domain:
            bounds[i] = (None, None)

        optimization = Optimization(
            LBFGS_NAME,
            wrapper_objective_voi,
            bounds,
            wrapper_gradient_voi,
            minimize=False)

        if self.opt_separing_domain:
            start = np.array(start)[:, self.bq.x_domain]
            starts = {}

            max_values = []
            for i, w in enumerate(self.domain_w[0]):
                starts[i] = np.concatenate((start[0, :], np.array([w])))

            args = (False, None, parallel, optimization, self, )
            opt_sol = Parallel.run_function_different_arguments_parallel(
                wrapper_optimization, starts, *args)

            for i in xrange(len(self.domain_w[0])):
                if opt_sol.get(i) is None:
                    logger.info("It wasn't possible to optimize for task %d" % i)
                    continue
                max_values.append(opt_sol[i]['optimal_value'])

            if len(max_values) == 0:
                raise Exception("Optimization failed for all the tasks!")
            index_max = np.argmax(max_values)
            results = opt_sol[index_max]
        else:
            results = optimization.optimize(start, *(self, ))

        self.optimization_results.append(results)

        return results

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
        self.max_mean = None

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed):
        """
        Write the results of the optimization.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
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
                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write(self.optimization_results, debug_path)

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
                values[task] = wrapper_evaluate_sbo(vectors, task, self)

        f_name = self._filename_voi_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)


        JSONFile.write({'points': points, 'evaluations': values}, debug_path)

        return values

