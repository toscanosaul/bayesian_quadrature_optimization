from __future__ import absolute_import

import unittest
import numpy.testing as npt

import numpy as np

from stratified_bayesian_optimization.lib.util_kernels import (
    find_define_kernel_from_array,
    find_kernel_constructor,
    define_prior_parameters_using_data,
)
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    LENGTH_SCALE_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    SIGMA2_NAME,
    LOWER_TRIANG_NAME,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel


class TestUtilKernels(unittest.TestCase):

    def test_find_define_kernel_from_array(self):
        assert Matern52.define_kernel_from_array == find_define_kernel_from_array(MATERN52_NAME)
        assert TasksKernel.define_kernel_from_array == find_define_kernel_from_array(
            TASKS_KERNEL_NAME)
        with self.assertRaises(NameError):
            find_define_kernel_from_array('a')

    def test_find_kernel_constructor(self):
        assert find_kernel_constructor(MATERN52_NAME) == Matern52
        assert find_kernel_constructor(TASKS_KERNEL_NAME) == TasksKernel

        with self.assertRaises(NameError):
            find_kernel_constructor('a')

    def test_define_prior_parameters_using_data(self):
        data = {
            'points': np.array([[1], [2]]),
            'evaluations': np.array([1, 2]),
            'var_noise': None,
        }
        type_kernel = [MATERN52_NAME]
        dimensions = [1]

        priors = define_prior_parameters_using_data(data, type_kernel, dimensions)
        assert len(priors) == 3
        assert priors[LOWER_TRIANG_NAME] is None
        assert priors[SIGMA2_NAME] == 0.25
        npt.assert_almost_equal(priors[LENGTH_SCALE_NAME], 1.5432098765432098)

        data2 = {
            'points': np.array([[0, 1], [0, 2]]),
            'evaluations': np.array([1, 2]),
            'var_noise': None,
        }
        dimensions = [2, 1, 1]
        type_kernel = [PRODUCT_KERNELS_SEPARABLE, TASKS_KERNEL_NAME, MATERN52_NAME]
        priors2 = define_prior_parameters_using_data(data2, type_kernel, dimensions)
        assert len(priors2) == 3
        npt.assert_almost_equal(priors2[LOWER_TRIANG_NAME], [-0.34657359027997259])
        assert priors2[SIGMA2_NAME] == 0.25
        npt.assert_almost_equal(priors2[LENGTH_SCALE_NAME], 1.5432098765432098)

        data3 = {
            'points': np.array([[1, 0], [2, 0]]),
            'evaluations': np.array([1, 2]),
            'var_noise': None,
        }
        dimensions = [2, 1, 1]
        type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
        priors3 = define_prior_parameters_using_data(data3, type_kernel, dimensions)
        assert len(priors3) == 3
        npt.assert_almost_equal(priors3[LOWER_TRIANG_NAME], [-0.34657359027997259])
        assert priors3[SIGMA2_NAME] == 0.25
        npt.assert_almost_equal(priors3[LENGTH_SCALE_NAME], 1.5432098765432098)

        data4 = {
            'points': np.array([[0], [0]]),
            'evaluations': np.array([1, 2]),
            'var_noise': None,
        }
        dimensions = [1]
        type_kernel = [TASKS_KERNEL_NAME]
        priors4 = define_prior_parameters_using_data(data4, type_kernel, dimensions)
        assert len(priors4) == 3
        assert priors4[SIGMA2_NAME] is None
        assert priors4[LENGTH_SCALE_NAME] is None
        npt.assert_almost_equal(priors4[LOWER_TRIANG_NAME], [-0.34657359027997259])


