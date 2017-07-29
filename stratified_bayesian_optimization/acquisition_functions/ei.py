from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from scipy.stats import norm
from random import shuffle

from copy import deepcopy

import itertools


from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    wrapper_optimize,
    wrapper_objective_acquisition_function,
    wrapper_gradient_acquisition_function,
)
from stratified_bayesian_optimization.lib.constant import (
    LBFGS_NAME,
    DEBUGGING_DIR,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)

#TODO: FINISH AND TEST EI

class EI(object):

    _filename = 'opt_ei_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}.json'.format


    _filename_ei_evaluations = '{iteration}_ei_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}' \
                               '.json'.format

    _filename_points_ei_evaluations = 'points_for_ei_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format

    def __init__(self, gp, noisy_evaluations=False):
        """

        :param gp: GP-model instance
        :param noisy_evaluations: (boolean)
        """

        self.gp = gp
        self.best_solution = None
        self.noisy_evaluations = noisy_evaluations
        self.optimization_results = []

    def evaluate(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Compute the EI acquisition function.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: float
        """

        post_parameters = self.gp.compute_posterior_parameters(
            point, var_noise, mean, parameters_kernel)

        mu = post_parameters['mean']
        cov = post_parameters['cov']

        cov = np.clip(cov, 0, None)

        best = self.gp.get_historical_best_solution(
            var_noise, mean, parameters_kernel, self.noisy_evaluations)

        normalized_factor = (mu - best) / np.sqrt(cov)
        first_term = (mu - best) * norm.cdf(normalized_factor)

        second_term = np.sqrt(cov) * norm.pdf(normalized_factor)

        evaluation = first_term + second_term

        return evaluation[0, 0]

    def evaluate_gradient(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Computes the gradient of EI.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :return: np.array(n)
        """

        post_parameters = self.gp.compute_posterior_parameters(
            point, var_noise, mean, parameters_kernel)

        mu = post_parameters['mean']
        cov = post_parameters['cov']

        cov = np.clip(cov, 0, None)

        best = self.gp.get_historical_best_solution(
            var_noise, mean, parameters_kernel, self.noisy_evaluations)

        std = np.sqrt(cov)

        gradient = self.gp.gradient_posterior_parameters(point, var_noise, mean, parameters_kernel)
        grad_mu = gradient['mean']
        grad_cov = gradient['cov']

        grad_std = 0.5 * grad_cov / np.sqrt(cov)

        grad_factor = (grad_mu * std - grad_std * (mu - best)) / cov
        normalized_factor = (mu - best) / std

        first_term = grad_mu * norm.cdf(normalized_factor) + \
                     (mu - best) * grad_factor * norm.pdf(normalized_factor)

        second_term = grad_std * norm.pdf(normalized_factor) - \
                      std * norm.pdf(normalized_factor) * grad_factor * normalized_factor

        gradient = first_term + second_term
        return gradient[0, :]


    def optimize(self, start=None, random_seed=None, parallel=True, n_restarts=10):
        """
        Optimizes EI

        :param start: np.array(n)
        :param random_seed: int
        :param parallel: boolean
        :param n_restarts: int
        :return:
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        bounds = self.gp.bounds

        if start is None:
            if 1 == self.gp.type_bounds[-1]:
                tasks = self.gp.bounds[-1]
                n_tasks = len(tasks)
                n_restarts = int(np.ceil(n_restarts / n_tasks) * n_tasks)

                ind = [[i] for i in range(n_restarts)]
                np.random.shuffle(ind)
                task_chosen = np.zeros((n_restarts, 1))
                n_task_per_group = n_restarts / n_tasks

                for i in xrange(n_tasks):
                    for j in xrange(n_task_per_group):
                        tk = ind[j + i * n_task_per_group]
                        task_chosen[tk, 0] = i

                start_points = DomainService.get_points_domain(
                    n_restarts, bounds[0: -1], type_bounds=self.gp.type_bounds[0: -1])

                start_points = np.concatenate((start_points, task_chosen), axis=1)
            else:
                start_points = DomainService.get_points_domain(
                    n_restarts, bounds, type_bounds=self.gp.type_bounds)

            start = np.array(start_points)

        bounds = [tuple(bound) for bound in bounds]

        objective_function = wrapper_objective_acquisition_function
        grad_function = wrapper_gradient_acquisition_function

        optimization = Optimization(
            LBFGS_NAME,
            objective_function,
            bounds,
            grad_function,
            minimize=False)

        point_dict = {}
        for j in xrange(n_restarts):
            point_dict[j] = start[j, :]

        args = (False, None, parallel, optimization, self)

        optimal_solutions = Parallel.run_function_different_arguments_parallel(
            wrapper_optimize, point_dict, *args)

        maximum_values = []
        for j in xrange(n_restarts):
            maximum_values.append(optimal_solutions.get(j)['optimal_value'])

        ind_max = np.argmax(maximum_values)

        logger.info("Results of the optimization of the EI: ")
        logger.info(optimal_solutions.get(ind_max))

        self.optimization_results.append(optimal_solutions.get(ind_max))

        return optimal_solutions.get(ind_max)

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
        for kernel in self.gp.type_kernel:
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
                             random_seed, iteration, n_points_by_dimension=None, n_tasks=0):
        """
        Generates evaluations of SBO, and write them in the debug directory.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param iteration: (int)
        :param n_points_by_dimension: [int] Number of points by dimension
        :param n_tasks: (int) n_tasks > 0 if the last element of the domain is a task

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


        f_name = self._filename_points_ei_evaluations(
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
            n_points = n_points_by_dimension
            if n_points is None:
                n_points = (bounds[0][1] - bounds[0][0]) * 10

            if n_tasks > 0:
                bounds_x = [bounds[i] for i in xrange(len(bounds) - 1)]
                n_points_x = [n_points[i] for i in xrange(len(n_points))]
            else:
                n_points_x = n_points
                bounds_x = bounds

            points = []
            for bound, number_points in zip(bounds_x, n_points_x):
                points.append(np.linspace(bound[0], bound[1], number_points))

            vectors = []
            for point in itertools.product(*points):
                vectors.append(point)

            JSONFile.write(vectors, debug_path)

        n = len(vectors)
        points_ = deepcopy(vectors)

        vectors = np.array(vectors)

        if n_tasks > 0:
            vectors_ = None
            for i in xrange(n_tasks):
                task_vector = np.zeros(n) + i
                task_vector = task_vector.reshape((n, 1))
                points_ = np.concatenate((vectors, task_vector), axis=1)

                if vectors_ is not None:
                    vectors_ = np.concatenate((vectors_, points_), axis=0)
                else:
                    vectors_ = points_
            vectors = vectors_


        # TODO: extend to the case where w can be continuous

        n = vectors.shape[0]

        points = {}
        for i in xrange(n):
            points[i] = vectors[i, :]

        args = (False, None, False, self,)
        val = Parallel.run_function_different_arguments_parallel(
            wrapper_objective_acquisition_function, points, *args)

        values = np.zeros(n)
        for i in xrange(n):
            values[i] = val.get(i)



        f_name = self._filename_ei_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write({'points': points_, 'evaluations': values}, debug_path)

        return values

