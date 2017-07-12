from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.models.gp_fitting_gaussian import (
    GPFittingGaussian,
)
from stratified_bayesian_optimization.lib.expectations import (
    gradient_uniform_finite,
)
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature

class TestExpectations(unittest.TestCase):

    def setUp(self):
        training_data_complex = {
            "evaluations": [1.0, 1.1],
            "points": [[42.2851784656, 0], [42.3851784656, 1]],
            "var_noise": []}

        self.complex_gp_2 = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_complex, [2, 1, 2])

        self.gp = BayesianQuadrature(self.complex_gp_2, [0], UNIFORM_FINITE, {TASKS: 2})

    def test_gradient_uniform_finite(self):
        f = self.gp.gp.evaluate_grad_cross_cov_respect_point
        points_1 = np.array([[41.0, 0]])
        point_3 = np.array([[41.0, 1]])
        points_2 = self.gp.gp.data['points']

        parameters_kernel = self.gp.gp.kernel.hypers_values_as_array
        a = f(points_1, points_2, parameters_kernel)[:, 0]
        b = f(point_3, points_2, parameters_kernel)[:, 0]
        gradient = (a+b)/2.0

        grad = gradient_uniform_finite(f, np.array([[41.0]]), [0], np.array([[0], [1]]), [1],
                                       points_2, parameters_kernel)
        grad = grad.reshape(points_2.shape[0])

        assert np.all(grad == gradient)
