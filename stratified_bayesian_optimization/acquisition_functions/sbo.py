from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from copy import deepcopy

import multiprocessing as mp

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
    wrapper_optimize,
    wrapper_evaluate_sample,
    wrapper_evaluate_gradient_sample,
    wrapper_evaluate_sbo_mc,
)
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.util import wrapper_evaluate_sbo
from stratified_bayesian_optimization.acquisition_functions.ei import EI


logger = SBOLog(__name__)


class SBO(object):

    _filename = 'opt_sbo_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}_mc_{monte_carlo}.json'.format

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
        if self.bq.separate_tasks:
            self.bounds_opt.append([None, None])
        self.opt_separing_domain = False

        # Bounds or list of number of points of the domain
        self.domain_w = [self.bq.gp.bounds[i] for i in self.bq.w_domain]

        if self.bq.distribution == UNIFORM_FINITE:
            self.opt_separing_domain = True

        self.optimization_results = []

        # Cached data for MC SBO
        self.samples = {} # Samples from a standard Gaussian r.v. used to estimate SBO

        # 'optimum': arg_max{a_n(x) + sigma(x, candidate_point)*sample}
        # 'max': max{a_n(x) + sigma(x, candidate_point)*sample}
        self.optimal_samples = {}



    def evaluate_sample(self, point, candidate_point, sample, var_noise=None, mean=None,
                        parameters_kernel=None, cache=True, n_threads=0):
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
            parameters_kernel=parameters_kernel, cache=cache, n_threads=n_threads)
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

    def evaluate_sbo_by_sample(self, candidate_point, sample, start=None,
                               var_noise=None, mean=None, parameters_kernel=None, n_restarts=5,
                               parallel=True, n_threads=0, **opt_params_mc):
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
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'max': float, 'optimum': np.array(n)}
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
            parallel = False

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

        arg_max = results_opt[np.argmax(solutions)]['solution']
        return {'max': np.max(solutions), 'optimum': arg_max}

    def evaluate_mc(self, candidate_point,  n_samples, var_noise=None, mean=None,
                    parameters_kernel=None, random_seed=None, parallel=True, n_restarts=10,
                    n_threads=0, **opt_params_mc):
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
        :param n_threads: (int)
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'value': float, 'std': float}
        """


        if random_seed is not None:
            np.random.seed(random_seed)

        bounds_x = [self.bounds_opt[i] for i in xrange(len(self.bounds_opt)) if i in
                 self.bq.x_domain]

        if self.bq.max_mean is not None:
            max_mean = self.bq.max_mean
        else:
            max_mean = self.bq.optimize_posterior_mean(
                random_seed=random_seed, n_treads=n_threads)['optimal_value']

        if tuple(candidate_point[0, :]) in self.samples:
            samples = self.samples[tuple(candidate_point[0, :])]
        else:
            samples = np.random.normal(0, 1, n_samples)
            self.samples = {}
            self.samples[tuple(candidate_point[0, :])] = samples

        max_values = []

        if tuple(candidate_point[0, :]) in self.optimal_samples:
            optimal_values = self.optimal_samples[tuple(candidate_point[0, :])]['max'].values()
            return {'value': np.mean(optimal_values) - max_mean,
                    'std': np.std(optimal_values) / n_samples}

#        self.optimal_samples = {}
        self.optimal_samples[tuple(candidate_point[0, :])] = {}
        self.optimal_samples[tuple(candidate_point[0, :])]['max'] = {}
        self.optimal_samples[tuple(candidate_point[0, :])]['optimum'] = {}

        if parallel:
            # Cache this computation, so we don't have to do it over and over again
            self.bq.get_parameters_for_samples(True, candidate_point, parameters_kernel, var_noise,
                                               mean)

            point_dict = {}
            for i in xrange(n_samples):
                start_points = DomainService.get_points_domain(n_restarts + 1, bounds_x,
                                                               type_bounds=len(bounds_x) * [0])
                if len(self.bq.optimal_solutions) > 0:
                    start = self.bq.optimal_solutions[-1]['solution']

                    start = [start] + start_points[0: -1]
                    start = np.array(start)
                else:
                    start = np.array(start_points)

                for j in xrange(n_restarts + 1):
                    point_dict[(j, i)] = [deepcopy(start[j:j+1,:]), samples[i]]
            args = (False, None, True, n_threads, self, candidate_point, var_noise, mean,
                    parameters_kernel, n_threads)

            simulated_values = Parallel.run_function_different_arguments_parallel(
                wrapper_evaluate_sbo_by_sample, point_dict, *args, **opt_params_mc)

            for i in xrange(n_samples):
                values = []
                for j in xrange(n_restarts + 1):
                    if simulated_values.get((j, i)) is None:
                        logger.info("Error in computing simulated value at sample %d" % i)
                        continue
                    values.append(simulated_values[(j, i)]['max'])
                maximum = simulated_values[(np.argmax(values), i)]['optimum']
                max_ = np.max(values)
                self.optimal_samples[tuple(candidate_point[0, :])]['max'][i] = max_
                self.optimal_samples[tuple(candidate_point[0, :])]['optimum'][i] = maximum
                max_values.append(max_)
        else:
            for i in xrange(n_samples):
                max_value = self.evaluate_sbo_by_sample(
                    candidate_point, samples[i], start=None, var_noise=var_noise, mean=mean,
                    parameters_kernel=parameters_kernel, n_restarts=n_restarts, parallel=True,
                    **opt_params_mc)
                max_values.append(max_value['max'])
                maximum = max_value['optimum']
                self.optimal_samples[tuple(candidate_point[0, :])]['max'][i] = max_value['max']
                self.optimal_samples[tuple(candidate_point[0, :])]['optimum'][i] = maximum

        return {'value': np.mean(max_values) - max_mean, 'std': np.std(max_values) / n_samples}

    def gradient_mc(self, candidate_point, parameters_kernel=None, var_noise=None,
                    mean=None, n_samples=None, random_seed=None, parallel=True, n_restarts=10,
                    n_threads=0, **opt_params_mc):
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
        :param n_threads: int
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: {'gradient': np.array(n), 'std': np.array(n)}
        """
        if tuple(candidate_point[0, :]) not in self.optimal_samples:
            self.evaluate_mc(candidate_point, n_samples, var_noise=var_noise, mean=mean,
                             parameters_kernel=parameters_kernel, random_seed=random_seed,
                             parallel=parallel, n_restarts=n_restarts, n_threads=n_threads,
                             **opt_params_mc)

        max_points = self.optimal_samples[tuple(candidate_point[0, :])]['optimum']


        samples = self.samples[tuple(candidate_point[0, :])]
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
        :param n_threads: (int)

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

    def objective_voi(self, point, monte_carlo=False, n_samples=1, n_restarts=1, n_threads=0,
                      **opt_params_mc):
        """
        Evaluates the VOI at point.
        :param point: np.array(n)
        :param monte_carlo: (boolean) If True, estimates the function by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_threads: (int)
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: float
        """

        point = point.reshape((1, len(point)))

        if not monte_carlo:
            value = self.evaluate(point, n_threads=n_threads)
        else:
            value = self.evaluate_mc(point, n_samples=n_samples, n_restarts=n_restarts,
                                     n_threads=n_threads, **opt_params_mc)['value']
        return value

    def grad_obj_voi(self, point, monte_carlo=False, n_samples=1, n_restarts=1, n_threads=0,
                     **opt_params_mc):
        """
        Evaluates the gradient of VOI at point.
        :param point: np.array(n)
        :param monte_carlo: (boolean) If True, estimates the function by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_threads: (int)
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int
        :return: np.array(n)
        """

        point = point.reshape((1, len(point)))

        if not monte_carlo:
            grad = self.evaluate_gradient(point, n_threads=n_threads)
        else:
            grad = self.gradient_mc(point, n_samples=n_samples, n_restarts=n_restarts,
                                    n_threads=n_threads, **opt_params_mc)['gradient']

        return grad

    def optimize(self, start=None, random_seed=None, parallel=True, monte_carlo=False, n_samples=1,
                 n_restarts_mc=1, n_restarts=10, start_ei=False, **opt_params_mc):
        """
        Optimizes the VOI.
        :param start: np.array(n)
        :param random_seed: int
        :param parallel: (boolean) For several tasks, it's run in paralle if it's True
        :param monte_carlo: (boolean) If True, estimates the objective function and gradient by MC.
        :param n_samples: (int) Number of samples for the MC method.
        :param n_restarts_mc: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_restarts: (int)
        :param start_ei: (boolean) If True, we choose starting points using EI.
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int

        :return: dictionary with the results of the optimization.
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        bounds = self.bq.bounds

        if start_ei:
            ei = EI(self.bq.gp)
            opt_ei = ei.optimize(n_restarts=100, parallel=parallel)
            st_ei = opt_ei['solution']
            st_ei = st_ei.reshape((1, len(st_ei)))
            print "ver"
            print st_ei
            n_restarts -= 1

        if start is None:
            if self.bq.separate_tasks and n_restarts > 0:
                tasks = self.bq.tasks
                n_tasks = len(tasks)

                n_restarts = int(np.ceil(float(n_restarts) / n_tasks) * n_tasks)

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
            elif n_restarts > 0:
                start_points = DomainService.get_points_domain(
                    n_restarts, bounds, type_bounds=self.bq.type_bounds)

            if n_restarts > 0:
                start = np.array(start_points)
                if start_ei:
                    start = np.concatenate((start, st_ei), axis=0)
            else:
                start = st_ei

            n_restarts = start.shape[0]
        else:
            n_restarts = 1


        print "cool"
        print start

        bounds = [tuple(bound) for bound in self.bounds_opt]

        # for i in self.bq.w_domain:
        #     bounds[i] = (None, None)

        optimization = Optimization(
            LBFGS_NAME,
            wrapper_objective_voi,
            bounds,
            wrapper_gradient_voi,
            minimize=False)

        point_dict = {}
        for j in xrange(n_restarts):
            point_dict[j] = start[j, :]

        n_jobs = min(n_restarts, mp.cpu_count())
        n_threads = max(int((mp.cpu_count() - n_jobs) / n_jobs), 1)

        args = (False, None, parallel, 0, optimization, self, monte_carlo, n_samples,
                    n_restarts_mc, opt_params_mc, n_threads)

        optimal_solutions = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, point_dict, *args)

        maximum_values = []
        for j in xrange(n_restarts):
            maximum_values.append(optimal_solutions.get(j)['optimal_value'])

        ind_max = np.argmax(maximum_values)

        logger.info("Results of the optimization of the SBO: ")
        logger.info(optimal_solutions.get(ind_max))

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
        self.samples = {}
        self.optimal_samples = {}

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         monte_carlo=False):
        """
        Write the results of the optimization.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param monte_carlo: (boolean)
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
                                monte_carlo=monte_carlo)

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

