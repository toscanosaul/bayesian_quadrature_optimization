from __future__ import absolute_import

import unittest
import numpy.testing as npt

from mock import mock_open, patch, MagicMock

import numpy as np

from stratified_bayesian_optimization.entities.objective import Objective


class TestObjective(unittest.TestCase):

    def setUp(self):
        self.problem_name = 'test_problem_noise'
        self.training_name = 'test_noise'
        self.random_seed = 1
        self.n_training = 1
        self.n_samples = 100
        self.noise = True
        self.obj = Objective(self.problem_name, self.training_name, self.random_seed, self.n_training,
                             self.n_samples, self.noise)

    def test_add_point(self):
        np.random.seed(1)
        val = self.obj.add_point(np.array([1.0]), [0.5])
        assert val == self.obj.objective_values[0]
        assert self.obj.evaluated_points == [[1.0]]
        assert self.obj.model_objective_values == [[0.5]]
        npt.assert_almost_equal(self.obj.objective_values, [1.0], decimal=1)
        assert self.obj.standard_deviation_evaluations == [7.8350152288466661e-05]

    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_builder(self, mock_mkdir, mock_exists):
        mock_exists.return_value = False
        obj = Objective(self.problem_name, self.training_name, self.random_seed,
                        self.n_training, self.n_samples, self.noise)

        mock_mkdir.assert_called_with('problems/test_problem_noise/partial_results')
        np.random.seed(1)
        val = obj.add_point(np.array([1.0]), [0.5])
        assert val == obj.objective_values[0]
        assert obj.evaluated_points == [[1.0]]
        assert obj.model_objective_values == [[0.5]]
        npt.assert_almost_equal(obj.objective_values, [1.0], decimal=1)
        assert obj.standard_deviation_evaluations == [7.8350152288466661e-05]


