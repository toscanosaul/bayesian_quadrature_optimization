from __future__ import absolute_import

from os import path
import os

from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    PARTIAL_RESULTS,
    PROBLEM_DIR,
    SBO_METHOD,
    MULTI_TASK_METHOD,
)


class Objective(object):
    _filename = 'results_{problem_name}_{training_name}_{n_points}_{random_seed}_{method}_' \
                'samples_params_{n_samples_parameters}.json'.format

    def __init__(self, problem_name, training_name, random_seed, n_training, n_samples=None,
                 noise=False, method=SBO_METHOD, n_samples_parameters=0, objective_function=None,
                 training_function=None):
        """

        :param problem_name: (str)
        :param training_name: (str)
        :param random_seed: int
        :param n_training: int
        :param n_samples: (int) Take n_samples evaluations when we have noisy evaluations
        :param noise: boolean, true if the evaluations are noisy
        :param method: (str) bgo method
        :param n_samples_parameters: int
        """
        self.evaluated_points = []
        self.objective_values = []
        self.model_objective_values = []
        self.standard_deviation_evaluations = []

        self.noise = noise
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.n_training = n_training
        self.problem_name = problem_name
        self.training_name = training_name

        self.objective_function = objective_function
        self.training_function = training_function

        if objective_function is None:
            name_module = TrainingDataService.get_name_module(problem_name)
            self.module = __import__(name_module, globals(), locals(), -1)
        else:
            self.module = None

        self.method = method
        self.n_samples_parameters = n_samples_parameters

        dir = path.join(PROBLEM_DIR, self.problem_name, PARTIAL_RESULTS)

        if not os.path.exists(dir):
            os.mkdir(dir)

        file_name = self._filename(
            problem_name=self.problem_name,
            training_name=self.training_name,
            n_points=self.n_training,
            random_seed=self.random_seed,
            method=self.method,
            n_samples_parameters=self.n_samples_parameters,
        )

        self.file_path = path.join(dir, file_name)

    def add_point(self, point, model_objective_value):
        """

        :param point: np.array(k)
        :param model_objective_value: float

        :return: float (optimal value)
        """

        self.evaluated_points.append(list(point))
        self.model_objective_values.append(model_objective_value)

        eval = self.evaluate_objective(
            self.module, list(point), n_samples=self.n_samples,
            objective_function=self.objective_function)
        self.objective_values.append(eval[0])

        if self.noise:
            self.standard_deviation_evaluations.append(eval[1])

        data = self.serialize()
        JSONFile.write(data, self.file_path)

        return eval[0]


    def serialize(self):
        return {
            'evaluated_points': self.evaluated_points,
            'objective_values': self.objective_values,
            'model_objective_values': self.model_objective_values,
            'standard_deviation_evaluations': self.standard_deviation_evaluations,
        }

    def set_data_from_file(self):
        data = JSONFile.read(self.file_path)

        if data is None:
            return

        self.evaluated_points = data['evaluated_points']
        self.objective_values = data['objective_values']
        self.model_objective_values = data['model_objective_values']
        self.standard_deviation_evaluations = data['standard_deviation_evaluations']

    @staticmethod
    def evaluate_objective(module, point, n_samples=None, objective_function=None):
        """
        Evalute the objective function.

        :param module:
        :param point: [float]
        :param n_samples: (int), number of samples used when the evaluations are noisy
        :return: float
        """

        if n_samples is None or n_samples == 0:
            if module is not None:
                return module.main_objective(point)
            else:
                return objective_function(point)
        else:
            if module is not None:
                return module.main_objective(n_samples, point)
            else:
                return objective_function(point)
