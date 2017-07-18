from __future__ import absolute_import

import numpy.testing as npt
import unittest

import numpy as np

from stratified_bayesian_optimization.lib.util_gp_fitting import (
    get_kernel_default,
    get_kernel_class,
    define_prior_parameters_using_data,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.lib.constant import (
    LENGTH_SCALE_NAME,
    SMALLEST_POSITIVE_NUMBER,
    LARGEST_NUMBER,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    LOWER_TRIANG_NAME,
    SIGMA2_NAME,
)


class TestUtilGPFitting(unittest.TestCase):

    def test_get_kernel_default(self):
        kernel_name = [MATERN52_NAME]
        dimension = [2]
        kernel = get_kernel_default(kernel_name, dimension)

        kernel_ = Matern52.define_kernel_from_array(2, np.ones(3))

        assert Matern52.compare_kernels(kernel, kernel_)

        kernel_name = [TASKS_KERNEL_NAME]
        dimension = [1]
        kernel = get_kernel_default(kernel_name, dimension)

        kernel_ = TasksKernel.define_kernel_from_array(1, np.array([0]))

        assert TasksKernel.compare_kernels(kernel, kernel_)

        kernel_name = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
        dimension = [2, 1, 1]
        kernel = get_kernel_default(kernel_name, dimension)

        kernel_ = ProductKernels.define_kernel_from_array([1, 1], [np.array([1]), np.array([0])],
                                                          [MATERN52_NAME, TASKS_KERNEL_NAME])
        assert ProductKernels.compare_kernels(kernel, kernel_)

        kernel = get_kernel_default(kernel_name, dimension, default_values=np.array([1, 0]))
        assert ProductKernels.compare_kernels(kernel, kernel_)
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.max == [LARGEST_NUMBER]

        compare = kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.min
        assert compare == SMALLEST_POSITIVE_NUMBER

        kernel = get_kernel_default(kernel_name, dimension, default_values=np.array([1, 0]),
                                    bounds=[[-1, 2]])

        assert ProductKernels.compare_kernels(kernel, kernel_)
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.max == 60.0

        compare = kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.min
        assert compare == SMALLEST_POSITIVE_NUMBER

    def test_get_kernel_class(self):
        assert get_kernel_class(MATERN52_NAME) == Matern52
        assert get_kernel_class(TASKS_KERNEL_NAME) == TasksKernel
        assert get_kernel_class(PRODUCT_KERNELS_SEPARABLE) == ProductKernels

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
        assert priors[SIGMA2_NAME] is None
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
        assert priors2[SIGMA2_NAME] is None
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
        assert priors3[SIGMA2_NAME] is None
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
