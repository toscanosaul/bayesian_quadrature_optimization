from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.lib.util_gp_fitting import *
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.lib.constant import (
    LENGTH_SCALE_NAME,
    SMALLEST_POSITIVE_NUMBER,
    LARGEST_NUMBER,
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

        kernel_ = ProductKernels.define_kernel_from_array([1, 1], [np.array([1, 1]), np.array([0])],
                                                          [MATERN52_NAME, TASKS_KERNEL_NAME])
        assert ProductKernels.compare_kernels(kernel, kernel_)

        kernel = get_kernel_default(kernel_name, dimension, default_values=np.array([1, 1, 0]))
        assert ProductKernels.compare_kernels(kernel, kernel_)
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.max == [LARGEST_NUMBER]
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.min == \
               SMALLEST_POSITIVE_NUMBER

        kernel = get_kernel_default(kernel_name, dimension, default_values=np.array([1, 1, 0]),
                                    bounds=[[-1, 2]])

        assert ProductKernels.compare_kernels(kernel, kernel_)
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.max == 9.25925925925926
        assert kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].prior.min == \
               SMALLEST_POSITIVE_NUMBER

    def test_get_kernel_class(self):
        assert get_kernel_class(MATERN52_NAME) == Matern52
        assert get_kernel_class(TASKS_KERNEL_NAME) == TasksKernel
        assert get_kernel_class(PRODUCT_KERNELS_SEPARABLE) == ProductKernels


