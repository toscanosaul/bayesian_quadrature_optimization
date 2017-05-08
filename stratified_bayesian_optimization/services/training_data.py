from __future__ import absolute_import

from os import path

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import (
    PROBLEM_DIR,
    FILE_PROBLEM,
)
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)

class TrainingDataService(object):
    _filename = 'training_data_{problem_name}_{n_training}.json'.format
    _data_dir = '{problem_name}/data'.format

    @classmethod
    def get_training_data(cls, problem_name, points, noise=False, n_samples=None):
        """

        :param problem_name:
        :param points: [[float]]
        :param noise: boolean, true if the evaluations are noisy
        :param n_samples: int. If noise is true, we take n_samples of the function to estimate it.
        :return: {'points': [[float]], 'evaluations': [float], 'var_noise': [float] or None}
        """

        n_training = len(points)

        file_name = cls._filename(
            problem_name=problem_name,
            n_training=n_training,
        )

        training_dir = path.join(PROBLEM_DIR, problem_name, 'data')
        training_path = path.join(training_dir, file_name)

        training_data = JSONFile.read(training_path)
        if training_data is not None:
            return training_data

        name_module = cls.get_name_module(problem_name)
        module = __import__(name_module , globals(), locals(), -1)

        training_data = {}
        training_data['points'] = []
        training_data['evaluations'] = []
        training_data['var_noise'] = []

        for point in points:
            training_data['points'].append(point)

            if noise:
                evaluation = cls.evaluate_function(module, n_samples, *point)
                training_data['var_noise'].append(evaluation[1])
            else:
                evaluation = cls.evaluate_function(module, *point)
            training_data['evaluations'].append(evaluation[0])


        JSONFile.write(training_data, training_path)

        return training_data

    @classmethod
    def get_name_module(cls, problem_name):
        name = PROBLEM_DIR + '.' + problem_name + '.' + FILE_PROBLEM
        return name

    @classmethod
    def evaluate_function(cls, module, n_samples, params):
        if n_samples is None:
            return module.main(*params)
        else:
            return module.main(n_samples, *params)
