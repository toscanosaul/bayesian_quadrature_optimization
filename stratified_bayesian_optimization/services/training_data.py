from __future__ import absolute_import

from os import path
import os

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
    _filename = 'training_data_{problem_name}_{training_name}_{n_points}_{random_seed}.json'.format
    _filename_domain = 'training_points_{problem_name}_{training_name}_{n_points}_' \
                       '{random_seed}.json'.format

    @classmethod
    def from_dict(cls, spec):
        """
        Create training data from dict

        :param spec: dict
        :return: {'points': [[float]], 'evaluations': [float], 'var_noise': [float] or None}
        """

        entry = {
            'problem_name': spec.get('problem_name'),
            'training_name': spec.get('training_name'),
            'bounds_domain': spec.get('bounds_domain'),
            'n_training': spec.get('n_training'),
            'points': spec.get('points'),
            'noise': spec.get('noise'),
            'n_samples': spec.get('n_samples'),
            'random_seed': spec.get('random_seed'),
            'parallel': spec.get('parallel'),
            'type_bounds': spec.get('type_bounds'),
        }

        return cls.get_training_data(**entry)

    @classmethod
    def get_training_data(cls, problem_name, training_name, bounds_domain, n_training=5,
                          points=None, noise=False, n_samples=None,
                          random_seed=DEFAULT_RANDOM_SEED, parallel=True, type_bounds=None,
                          cache=True, gp_path_cache=None, simplex_domain=None,
                          objective_function=None):
        """

        :param problem_name: str
        :param training_name: (str), prefix used to save the training data.
        :param bounds_domain: [([float, float] or [float])], the first case is when the bounds are
            lower or upper bound of the respective entry; in the second case, it's list of finite
            points representing the domain of that entry.
        :param n_training: (int), number of training points if points is None
        :param points: [[float]]
        :param noise: boolean, true if the evaluations are noisy
        :param n_samples: int. If noise is true, we take n_samples of the function to estimate its
            value.
        :param random_seed: int
        :param parallel: (boolean) Train in parallel if it's True.
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param cache: (boolean) Try to get model from cache
        :return: {'points': [[float]], 'evaluations': [float], 'var_noise': [float] or []}
        """

        if cache and gp_path_cache is not None:
            data = JSONFile.read(gp_path_cache)
            if data is not None:
                return data['data']

        logger.info("Getting training data")

        rs = random_seed
        if points is not None and len(points) > 0:
            n_training = len(points)
            rs = 0

        file_name = cls._filename(
            problem_name=problem_name,
            training_name=training_name,
            n_points=n_training,
            random_seed=rs,
        )

        if not os.path.exists(PROBLEM_DIR):
            os.mkdir(PROBLEM_DIR)

        training_dir = path.join(PROBLEM_DIR, problem_name, 'data')

        if not os.path.exists(path.join(PROBLEM_DIR, problem_name)):
            os.mkdir(path.join(PROBLEM_DIR, problem_name))

        if not os.path.exists(training_dir):
            os.mkdir(training_dir)

        training_path = path.join(training_dir, file_name)

        if cache:
            training_data = JSONFile.read(training_path)
        else:
            training_data = None

        if training_data is not None:
            return training_data

        if n_training == 0:
            return {'points': [], 'evaluations': [], 'var_noise': []}

        np.random.seed(random_seed)

        if points is None or len(points) == 0:
            points = cls.get_points_domain(n_training, bounds_domain, random_seed, training_name,
                                           problem_name, type_bounds, simplex_domain=simplex_domain)

        if objective_function is None:
            name_module = cls.get_name_module(problem_name)
            module = __import__(name_module, globals(), locals(), -1)
        else:
            name_module = None
            module = None

        training_data = {}
        training_data['points'] = points
        training_data['evaluations'] = []
        training_data['var_noise'] = []

        if not parallel:
            for point in points:
                if noise:
                    if module is not None:
                        evaluation = cls.evaluate_function(module, point, n_samples)
                    else:
                        evaluation = objective_function(point, n_samples)
                    training_data['var_noise'].append(evaluation[1])
                else:
                    if module is not None:
                        evaluation = cls.evaluate_function(module, point)
                    else:
                        evaluation = objective_function(point)
                training_data['evaluations'].append(evaluation[0])
                JSONFile.write(training_data, training_path)
            JSONFile.write(training_data, training_path)
            return training_data

        arguments = convert_list_to_dictionary(points)

        if name_module is not None:
            kwargs = {'name_module': name_module, 'cls_': cls, 'n_samples': n_samples}
        else:
            kwargs = {'name_module': None, 'cls_': cls, 'n_samples': n_samples,
                      'objective_function': objective_function}

        training_points = Parallel.run_function_different_arguments_parallel(
            wrapper_evaluate_objective_function, arguments, **kwargs)

        training_points = convert_dictionary_to_list(training_points)

        training_data['evaluations'] = [value[0] for value in training_points]

        if noise:
            training_data['var_noise'] = [value[1] for value in training_points]

        if cache:
            JSONFile.write(training_data, training_path)

        return training_data

    @classmethod
    def get_points_domain(cls, n_training, bounds_domain, random_seed, training_name, problem_name,
                          type_bounds=None, simplex_domain=None):
        """
        Get random points in the domain.

        :param n_training: (int) Number of points
        :param bounds_domain: [([float, float] or [float])], the first case is when the bounds are
            lower or upper bound of the respective entry; in the second case, it's list of finite
            points representing the domain of that entry.
        :param random_seed: (int)
        :param training_name: (str), prefix used to save the training data.
        :param problem_name: str
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :return: [[float]]
        """

        file_name = cls._filename_domain(
            problem_name=problem_name,
            training_name=training_name,
            n_points=n_training,
            random_seed=random_seed,
        )

        training_dir = path.join(PROBLEM_DIR, problem_name, 'data')
        training_path = path.join(training_dir, file_name)

        points = JSONFile.read(training_path)
        if points is not None:
            return points

        points = DomainService.get_points_domain(n_training, bounds_domain, type_bounds=type_bounds,
                                                 random_seed=random_seed,
                                                 simplex_domain=simplex_domain)
        print(points)
        JSONFile.write(points, training_path)

        return points

    @classmethod
    def get_name_module(cls, problem_name):
        name = PROBLEM_DIR + '.' + problem_name + '.' + FILE_PROBLEM
        return name

    @classmethod
    def evaluate_function(cls, module, point, n_samples=None):
        """
        Evalute the objective function.

        :param module:
        :param point: [float]
        :param n_samples: (int), number of samples used when the evaluations are noisy
        :return: float
        """

        if n_samples is None or n_samples == 0:
            return module.main(point)
        else:
            return module.main(n_samples, point)
