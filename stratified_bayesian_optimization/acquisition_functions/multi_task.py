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
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.util import (
    wrapper_objective_acquisition_function,
)

logger = SBOLog(__name__)


class MultiTasks(object):

    def __init__(self, bq, n_tasks):
        self.bq = bq # Don't forget that we should have that self.bq.model_only_x = True
        self.n_tasks = n_tasks
        self.ei_tasks = EI(self.bq.gp)
        self.ei = EI(self.bq)

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
                       n_samples_parameters=0, n_best_restarts=0):
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
                                    n_samples_parameters=n_samples_parameters)

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

        return np.argmax(values)

    def optimize(self, random_seed=None, parallel=True, n_restarts=100, n_best_restarts=0,
                 n_samples_parameters=0, start_new_chain=True, **kwargs):
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
                                    n_best_restarts=n_best_restarts)

        task = self.choose_best_task_given_x(point, n_samples_parameters=n_samples_parameters)

        solution = np.concatenate((point, np.array([task])))

        return {'solution': solution}

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         **kwargs):
        self.ei.write_debug_data(problem_name, model_type, training_name, n_training, random_seed)

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.ei_tasks.clean_cache()
        self.ei.clean_cache()
