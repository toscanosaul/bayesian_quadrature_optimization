from __future__ import absolute_import

import unittest

import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    SIGMA2_NAME,
)
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences


class TestScaledKernel(unittest.TestCase):

    def setUp(self):
        self.dimension = 2
        self.length_scale = ParameterEntity('scale', np.array([1.0, 2.0]), None)
        self.sigma2 = ParameterEntity('sigma2', np.array([3]), None)
        self.matern52 = Matern52(self.dimension, self.length_scale)
        self.matern52 = ScaledKernel(self.dimension, self.matern52, self.sigma2)

    def test_define_default_kernel(self):
        kernel = ScaledKernel.define_default_kernel(1, None, np.array([1.0, 1.0]), None,
                                                    *([MATERN52_NAME],))

        assert kernel.name == MATERN52_NAME
        assert kernel.dimension == 1
        assert kernel.dimension_parameters == 2
        assert kernel.sigma2.value == 1.0
        assert kernel.kernel.length_scale.value == 1.0

    def test_define_default_kernel_2(self):
        kernel = ScaledKernel.define_default_kernel(1, None, None, None, *([MATERN52_NAME],))
        assert kernel.name == MATERN52_NAME
        assert kernel.dimension == 1
        assert kernel.dimension_parameters == 2
        assert kernel.sigma2.value == [1.0]
        assert kernel.kernel.length_scale.value == [1.0]

    def test_define_prior_parameters(self):
        training_data = {
            "evaluations": np.array([1.0, 2.0]),
            "points": np.array([[1.0], [2.0]]),
            "var_noise": None}
        assert ScaledKernel.define_prior_parameters(training_data, 0, None) == {
            SIGMA2_NAME: 0.25,
        }

    def test_compare_kernels(self):
        kernel_t = TasksKernel(self.dimension, np.array([0.0]))
        assert ScaledKernel.compare_kernels(self.matern52, kernel_t) is False

        kernel_s = Matern52(3, self.length_scale)
        assert ScaledKernel.compare_kernels(self.matern52, kernel_s) is False

        kernel_s = Matern52(2, self.length_scale)
        assert ScaledKernel.compare_kernels(self.matern52, kernel_s) is False

        sigma2 = ParameterEntity('sigma2', np.array([1]), None)
        kernel = ScaledKernel(self.dimension, kernel_s, sigma2)
        assert ScaledKernel.compare_kernels(self.matern52, kernel) is False

        kernel_s = Matern52(2, ParameterEntity('scale', np.array([1.0, 3.0]), None))
        kernel = ScaledKernel(self.dimension, kernel_s, self.sigma2)
        assert ScaledKernel.compare_kernels(self.matern52, kernel) is False

    def test_evaluate_grad_respect_point(self):
        result = ScaledKernel.evaluate_grad_respect_point(np.array([5.0, 1.0]), np.array([[1]]),
                                                           np.array([[4], [5]]), 1,
                                                          *([MATERN52_NAME],))

        kernel = ScaledKernel.define_kernel_from_array(1, np.array([5.0, 1.0]), *([MATERN52_NAME],))
        assert np.all(result == kernel.grad_respect_point(np.array([[1]]), np.array([[4], [5]])))

    def test_evaluate_hessian_respect_point(self):
        point = np.array([[4.5, 7.5]])
        inputs = np.array([[5.0, 6.0], [8.0, 9.0]])
        params = np.array([1.0, 5.0, 3.0])
        result = ScaledKernel.evaluate_hessian_respect_point(
            params, point, inputs, 2, *([MATERN52_NAME],))


        dh = 0.00001
        finite_diff = FiniteDifferences.second_order_central(
            lambda x: ScaledKernel.evaluate_cross_cov_defined_by_params(
                params, x.reshape((1, len(x))), inputs, 2, *([MATERN52_NAME],)),
            point[0, :], np.array([dh])
        )


        for i in xrange(2):
            for j in xrange(2):
                npt.assert_almost_equal(finite_diff[i, j],
                                        np.array([[result[0, i, j], result[1, i, j]]]), decimal=5)

