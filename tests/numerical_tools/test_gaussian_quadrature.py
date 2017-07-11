import unittest

import numpy as np

from stratified_bayesian_optimization.models.gp_fitting_gaussian import (
    GPFittingGaussian,
)
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.numerical_tools.gaussian_quadrature import GaussianQuadrature


class TestGaussianQuadrature(unittest.TestCase):

    def setUp(self):
        self.training_data_complex = {
            "evaluations": [1.0, 1.1],
            "points": [[42.2851784656, 0], [42.3851784656, 0]],
            "var_noise": []}

        self.complex_gp = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            self.training_data_complex, [2, 1, 1])

        self.gp = GaussianQuadrature(self.complex_gp, [0], UNIFORM_FINITE, {TASKS: 1})

        training_data_complex = {
            "evaluations": [1.0, 1.1],
            "points": [[42.2851784656, 0], [42.3851784656, 1]],
            "var_noise": []}

        self.complex_gp_2 = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_complex, [3, 1, 2])

        self.gp_2 = GaussianQuadrature(self.complex_gp_2, [0], UNIFORM_FINITE, {TASKS: 2})


    def test_evaluate_quadrature_cross_cov(self):
        point = np.array([[1.0]])
        points_2 = np.array([[42.2851784656, 0], [42.3851784656, 0]])

        parameters_kernel = self.gp.gp.kernel.hypers_values_as_array
        value = self.gp.evaluate_quadrature_cross_cov(point, points_2, parameters_kernel)

        value_1 = self.gp.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                np.array([[42.2851784656, 0]]), parameters_kernel)
        value_2 = self.gp.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                np.array([[42.3851784656, 0]]), parameters_kernel)
        assert value[0] == value_1[0, 0]
        assert value[1] == value_2[0, 0]

        point = np.array([[1.0]])
        points_2 = np.array([[42.2851784656, 0], [42.3851784656, 1]])

        parameters_kernel = self.gp_2.gp.kernel.hypers_values_as_array
        value = self.gp_2.evaluate_quadrature_cross_cov(point, points_2, parameters_kernel)

        value_1 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                  np.array([[42.2851784656, 0]]), parameters_kernel)
        value_2 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 1.0]]),
                                                  np.array([[42.2851784656, 0]]), parameters_kernel)

        assert value[0] == np.mean([value_1, value_2])

        value_1 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                  np.array([[42.3851784656, 1]]), parameters_kernel)
        value_2 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 1.0]]),
                                                  np.array([[42.3851784656, 1]]), parameters_kernel)

        assert value[1] == np.mean([value_1, value_2])

    def test_compute_posterior_parameters(self):
        points = np.array([[42.0], [42.1], [1.0]])
        candidate_point = np.array([[41.0, 1.0]])
        value = self.gp_2.compute_posterior_parameters(points, candidate_point)

        points_ = np.array([[42.0, 0], [42.0, 1]])
        values = self.gp_2.gp.compute_posterior_parameters(points_)
        assert np.mean(values['mean']) == value['mean'][0]

        points_ = np.array([[1.0, 0], [1.0, 1]])
        values = self.gp_2.gp.compute_posterior_parameters(points_)
        assert np.mean(values['mean']) == value['mean'][2]

        print value
        assert 1 ==0
