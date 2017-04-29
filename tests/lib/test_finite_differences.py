from __future__ import absolute_import

import unittest

from doubles import expect

import numpy as np

from copy import deepcopy

from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.kernels.matern52 import Matern52


class TestFiniteDifferences(unittest.TestCase):

    def setUp(self):
        self.h = np.array([1.0])
        def f(x):
            return Matern52.evaluate_cov_defined_by_params(x, np.array([[2.0, 0.0], [0.0, 2.0]]), 2)
        self.f = f
        self.x = np.array([1.0, 1.0, 1.0])


    def test_forward_difference(self):

        result = FiniteDifferences.forward_difference(self.f, self.x, self.h)

        base_eval = \
            Matern52.evaluate_cov_defined_by_params(self.x, np.array([[2.0, 0.0], [0.0, 2.0]]), 2)

        for i in result:
            new_x = deepcopy(self.x)
            new_x[i] += self.h[0]
            new_eval = \
                Matern52.evaluate_cov_defined_by_params(
                    new_x, np.array([[2.0, 0.0], [0.0, 2.0]]), 2)
            for j in range(2):
                for h in range(2):
                    assert result[i][j, h] == (new_eval[j, h] - base_eval[j, h]) / self.h[0]
