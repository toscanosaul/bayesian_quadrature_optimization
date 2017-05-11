from __future__ import absolute_import

from os import path

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import (
    PROBLEM_DIR,
    FILE_PROBLEM,
    DEFAULT_RANDOM_SEED,
)
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    wrapper_evaluate_objective_function,
    convert_list_to_dictionary,
    convert_dictionary_to_list,
)

logger = SBOLog(__name__)

class TrainingDataService(object):
    _filename = 'training_data_{problem_name}_{training_name}.json'.format
    _filename_domain = 'training_points_{problem_name}_{training_name}_{n_points}_' \
                       '{random_seed}.json'.format

    @classmethod
    def get_training_data(cls, problem_name, training_name, points=None, n_training=0,
                          noise=False, n_samples=None, random_seed=DEFAULT_RANDOM_SEED,
                          parallel=True):
        """

        :param problem_name: str
        :param training_name: (str), prefix used to save the training data.
        :param points: [[float]]
        :param n_training: (int), number of training points if points is None
        :param noise: boolean, true if the evaluations are noisy
        :param n_samples: int. If noise is true, we take n_samples of the function to estimate its
            value.
        :param random_seed: int
        :param parallel: (boolean) Train in parallel if it's True.
        :return: {'points': [[float]], 'evaluations': [float], 'var_noise': [float] or None}
        """

        file_name = cls._filename(
            problem_name=problem_name,
            training_name=training_name,
        )

        training_dir = path.join(PROBLEM_DIR, problem_name, 'data')
        training_path = path.join(training_dir, file_name)

        training_data = JSONFile.read(training_path)
        if training_data is not None:
            return training_data

        np.random.seed(random_seed)

        if points is None:
            points = cls.get_points_domain(n_samples, random_seed, training_name, problem_name)


        name_module = cls.get_name_module(problem_name)
        module = __import__(name_module , globals(), locals(), -1)

        training_data = {}
        training_data['points'] = points
        training_data['evaluations'] = []
        training_data['var_noise'] = []

        if not parallel:
            for point in points:
                if noise:
                    evaluation = cls.evaluate_function(module, point, n_samples)
                    training_data['var_noise'].append(evaluation[1])
                else:
                    evaluation = cls.evaluate_function(module, point)
                training_data['evaluations'].append(evaluation[0])
            JSONFile.write(training_data, training_path)
            return training_data

        kwargs = {'n_samples':n_samples, 'module': module}

        arguments = convert_list_to_dictionary(points)
        training_points = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_objective_function, arguments, all_success=False, **kwargs)

        training_points = convert_dictionary_to_list(training_points)
        if noise:
            training_data['var_noise'] = [value[1] for value in training_points]
            training_data['evaluations'] = [value[0] for value in training_points]
        else:
            training_data['evaluations'] = [value for value in training_points]

        return training_data

    @classmethod
    def get_points_domain(cls, n_samples, random_seed, training_name, problem_name):
        """
        Get random points in the domain.

        :param n_samples: (int)
        :param random_seed: (int)
        :param training_name: (str), prefix used to save the training data.
        :param problem_name: str
        :return: [[float]]
        """

        file_name = cls._filename_domain(
            problem_name=problem_name,
            training_name=training_name,
            n_points=n_samples,
            random_seed=random_seed,
        )

        training_dir = path.join(PROBLEM_DIR, problem_name, 'data')
        training_path = path.join(training_dir, file_name)

        points = JSONFile.read(training_path)
        if points is not None:
            return points

        points = DomainService.get_points_domain(n_samples, random_seed)

        JSONFile.write(points, training_path)

        return points

    @classmethod
    def get_name_module(cls, problem_name):
        name = PROBLEM_DIR + '.' + problem_name + '.' + FILE_PROBLEM
        return name

    @classmethod
    def evaluate_function(cls, module, params, n_samples=None):
        """
        Evalute the objective function.

        :param module:
        :param params: [float]
        :param n_samples: (int), number of samples used when the evaluations are noisy
        :return: float
        """
        if n_samples is None:
            return module.main(params)
        else:
            return module.main(n_samples, params)
