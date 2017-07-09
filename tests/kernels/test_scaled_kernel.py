from __future__ import absolute_import

import unittest

from doubles import expect

import copy
import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.matern52 import Matern52, GradientLSMatern52
from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
    MATERN52_NAME,
    SMALLEST_POSITIVE_NUMBER,
    SIGMA2_NAME,
    LENGTH_SCALE_NAME,
)


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

    def test_define_default_kernel(self):
        kernel = ScaledKernel.define_default_kernel(1, None, None, None, *([MATERN52_NAME],))
        assert kernel.name == MATERN52_NAME
        assert kernel.dimension == 1
        assert kernel.dimension_parameters == 2
        assert kernel.sigma2.value == [1.0]
        assert kernel.kernel.length_scale.value == [1.0]

    def test_define_prior_parameters(self):
        training_data = {
            "evaluations":np.array([1.0, 2.0]),
            "points":np.array([[1.0], [2.0]]),
            "var_noise":None}
        assert ScaledKernel.define_prior_parameters(training_data, 0, None) == {
            SIGMA2_NAME: 0.25,
        }

    def test_compare_kernels(self):
        kernel_t = TasksKernel(self.dimension, np.array([0.0]))
        assert ScaledKernel.compare_kernels(self.matern52, kernel_t) is False

        kernel_s =  Matern52(3, self.length_scale)
        assert ScaledKernel.compare_kernels(self.matern52, kernel_s) is False

        kernel_s =  Matern52(2, self.length_scale)
        assert ScaledKernel.compare_kernels(self.matern52, kernel_s) is False

        sigma2 = ParameterEntity('sigma2', np.array([1]), None)
        kernel = ScaledKernel(self.dimension, kernel_s, sigma2)
        assert ScaledKernel.compare_kernels(self.matern52, kernel) is False

        kernel_s = Matern52(2, ParameterEntity('scale', np.array([1.0, 3.0]), None))
        kernel = ScaledKernel(self.dimension, kernel_s, self.sigma2)
        assert ScaledKernel.compare_kernels(self.matern52, kernel) is False
