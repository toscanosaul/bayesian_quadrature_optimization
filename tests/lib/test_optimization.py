from __future__ import absolute_import

import unittest

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.constant import LBFGS_NAME


class TestOptimization(unittest.TestCase):

    def setUp(self):
        def f(x):
            return x**2

        def grad(x):
            return 2.0 * x

        self.bounds = [(-1, 1)]

        self.opt = Optimization(LBFGS_NAME, f, self.bounds, grad)
        self.opt_2 = Optimization(LBFGS_NAME, f, self.bounds, grad, minimize=False)

    def test_get_optimizer(self):
        assert Optimization._get_optimizer(LBFGS_NAME) == fmin_l_bfgs_b

    def test_optimize(self):
        opt = self.opt.optimize(np.array([0.9]))
        assert opt['solution'] == 0
        assert opt['optimal_value'] == 0
        assert opt['gradient'] == 0

        opt_2 = self.opt_2.optimize(np.array([0.9]))
        assert opt_2['solution'] == 1
        assert opt_2['optimal_value'] == 1
        assert opt_2['gradient'] == 1
