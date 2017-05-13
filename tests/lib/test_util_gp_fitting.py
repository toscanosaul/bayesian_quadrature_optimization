from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.lib.util_gp_fitting import *
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels


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

    def test_get_kernel_class(self):
        assert get_kernel_class(MATERN52_NAME) == Matern52
        assert get_kernel_class(TASKS_KERNEL_NAME) == TasksKernel
        assert get_kernel_class(PRODUCT_KERNELS_SEPARABLE) == ProductKernels

