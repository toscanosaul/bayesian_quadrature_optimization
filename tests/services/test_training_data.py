import unittest
from doubles import expect

import numpy as np
import numpy.testing as npt
import os
from mock import patch, MagicMock

from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    DEFAULT_RANDOM_SEED,
)


class MockMkdir(object):
    def __init__(self):
        self.received_args = None

    def __call__(self, *args):
        self.received_args = args


class TestTrainingDataService(unittest.TestCase):

    def test_get_training_data(self):
        problem_name = 'test_problem'
        training_name = 'test'
        bounds_domain = [[1, 100]]
        expect(JSONFile).read.and_return(None)
        training_data = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain)

        np.random.seed(DEFAULT_RANDOM_SEED)
        points = \
            [[42.2851784656], [72.3121248508], [1.0113231069], [30.9309246906], [15.5288331909]]
        evaluations = [i[0] for i in points]

        assert training_data['var_noise'] == []
        npt.assert_almost_equal(training_data['evaluations'], evaluations)
        npt.assert_almost_equal(training_data['points'], points)

        training_data_ = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain,
                                                  parallel=False)

        assert training_data['var_noise'] == training_data_['var_noise']
        assert np.all(training_data['evaluations'] == training_data_['evaluations'])
        assert np.all(training_data['points'] == training_data_['points'])

        with patch('os.path.exists', new=MagicMock(return_value=False)):
            os.mkdir = MockMkdir()
            training_data_ = \
                TrainingDataService.get_training_data(problem_name, training_name, bounds_domain,
                                                      parallel=False)
            assert training_data['var_noise'] == training_data_['var_noise']
            assert np.all(training_data['evaluations'] == training_data_['evaluations'])
            assert np.all(training_data['points'] == training_data_['points'])

    def test_cached_get_training_data(self):
        problem_name = 'test_problem'
        training_name = 'test'
        bounds_domain = [[1, 100]]

        expect(JSONFile).read.and_return(0)
        training_data = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain)
        assert training_data == 0

    def test_get_training_data_noise(self):
        problem_name = 'test_problem_noise'
        training_name = 'test'
        bounds_domain = [[1, 100]]
        expect(JSONFile).read.and_return(None)
        training_data = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain,
                                                  n_training=1, noise=True, n_samples=5)
        np.random.seed(DEFAULT_RANDOM_SEED)
        points = [list(np.random.uniform(1, 100, 1))]
        noise = np.random.normal(0, 1, 5)
        eval = points[0] + noise
        evaluations = [np.mean(eval)]
        var = [np.var(eval) / 25.0]

        npt.assert_almost_equal(training_data['points'], points)
        npt.assert_almost_equal(training_data['var_noise'], var)
        npt.assert_almost_equal(training_data['evaluations'], evaluations)

        training_data_ = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain,
                                                  n_training=1, noise=True, n_samples=5,
                                                  parallel=False)

        assert np.all(training_data['points'] == training_data_['points'])
        assert np.all(training_data['var_noise'] == training_data_['var_noise'])
        assert np.all(training_data['evaluations'] == training_data_['evaluations'])

    def test_get_training_data_given_points(self):
        points = \
            [[42.2851784656], [72.3121248508], [1.0113231069], [30.9309246906], [15.5288331909]]
        problem_name = 'test_problem'
        training_name = 'test_given_points'
        bounds_domain = [[1, 100]]
        expect(JSONFile).read.and_return(None)
        training_data = \
            TrainingDataService.get_training_data(problem_name, training_name, bounds_domain,
                                                  points=points)

        assert training_data['var_noise'] == []
        assert np.all(training_data['evaluations'] == [i[0] for i in points])
        assert np.all(training_data['points'] == points)

    def test_get_training_data_cached_points(self):
        problem_name = 'test_problem'
        training_name = 'test'
        points = TrainingDataService.get_points_domain(5, [[1, 100]], DEFAULT_RANDOM_SEED,
                                                       training_name, problem_name)
        compare_point = \
            [[42.2851784656], [72.3121248508], [1.0113231069], [30.9309246906], [15.5288331909]]
        assert points == compare_point

    def test_training_data_from_dict(self):
        problem_name = 'test_problem'
        training_name = 'test'
        bounds_domain = [[1, 100]]
        expect(JSONFile).read.and_return(None)
        np.random.seed(DEFAULT_RANDOM_SEED)
        points = \
            [[42.2851784656], [72.3121248508], [1.0113231069], [30.9309246906], [15.5288331909]]

        dict = {
            'problem_name': problem_name,
            'training_name': training_name,
            'bounds_domain': bounds_domain,
            'n_training': 5,
            'points': points,
            'noise': False,
            'n_samples': 0,
            'random_seed': 1,
            'parallel': True,
            'type_bounds': [0],
        }

        training_data = TrainingDataService.from_dict(dict)

        len(training_data) == 3
        assert training_data['var_noise'] == []
        assert np.all(training_data['evaluations'] == [i[0] for i in points])
        assert np.all(training_data['points'] == points)
