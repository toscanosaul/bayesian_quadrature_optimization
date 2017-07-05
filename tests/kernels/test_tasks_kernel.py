from __future__ import absolute_import

import unittest

from doubles import expect

import numpy as np
import numpy.testing as npt
import copy

from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel, GradientTasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
    TASKS_KERNEL_NAME,
    LOWER_TRIANG_NAME,
)


class TestTasksKernel(unittest.TestCase):

    def setUp(self):
        self.n_tasks = 1
        self.prior_2 = UniformPrior(1, [1], [100])
        self.lower_triang = ParameterEntity('lower_triang', np.array([1.0]), self.prior_2)
        self.task_kernel = TasksKernel(self.n_tasks, self.lower_triang)
        self.inputs = np.array([[0]])
        self.inputs_ = np.array([[0], [1]])

    def test_hypers(self):
        assert self.task_kernel.dimension_parameters == 1
        assert {'lower_triang': self.lower_triang} == self.task_kernel.hypers

    def test_name_parameters_as_list(self):
        assert self.task_kernel.name_parameters_as_list == \
               [('lower_triang', [(0, None)])]

    def test_define_kernel_from_array(self):
        kernel = TasksKernel.define_kernel_from_array(5, np.array([1, 2, 3]))
        assert kernel.n_tasks == 5
        assert np.all(kernel.lower_triang.value == np.array([1, 2, 3]))

    def test_set_parameters(self):
        lower_traing = ParameterEntity('lw', np.array([3.0]), None)

        self.task_kernel.set_parameters(lower_traing)

        assert self.task_kernel.hypers == {'lw': lower_traing}
        assert self.task_kernel.chol_base_cov_matrix == np.array([[np.exp(3.0)]])
        assert self.task_kernel.base_cov_matrix == np.array([[np.exp(6.0)]])

        self.task_kernel.compute_cov_matrix()
        assert self.task_kernel.chol_base_cov_matrix == np.array([[np.exp(3.0)]])
        assert self.task_kernel.base_cov_matrix == np.array([[np.exp(6.0)]])

    def test_cov(self):
        expect(self.task_kernel).cross_cov.once().and_return(0)
        assert self.task_kernel.cov(self.inputs) == 0

    def test_cross_cov(self):
        npt.assert_almost_equal(self.task_kernel.cross_cov(self.inputs, self.inputs),
                                np.array([[np.exp(2.0)]]))

    def test_gradient_respect_parameters(self):
        expect(GradientTasksKernel).gradient_respect_parameters.once().and_return(
            {0: np.array([[0]])})

        gradient = {}
        gradient['lower_triang'] = {}
        gradient['lower_triang'][0] = np.array([[0]])

        assert self.task_kernel.gradient_respect_parameters(self.inputs) == gradient

    def test_gradient_respect_parameters_finite_differences(self):
        dh = 0.00000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda params: TasksKernel.evaluate_cov_defined_by_params(params, self.inputs_, 2),
            np.array([2.0, 3.0, 4.0]), np.array([dh]))

        gradient = TasksKernel.evaluate_grad_defined_by_params_respect_params(
            np.array([2.0, 3.0, 4.0]), self.inputs_, 2)

        for i in range(3):
            npt.assert_almost_equal(finite_diff[i], gradient[i],  decimal=4)

    def test_grad_respect_point(self):
        assert self.task_kernel.grad_respect_point(self.inputs, self.inputs) == np.array([[0]])

    def test_gradient_respect_parameters_gradient_class(self):
        grad = GradientTasksKernel.gradient_respect_parameters(np.array([[np.exp(1.0)]]), 1)
        assert len(grad) == 1
        npt.assert_almost_equal(grad[0], np.array([[2.0 * np.exp(2.0)]]))

    def test_evaluate_cov_defined_by_params(self):
        result = TasksKernel.evaluate_cov_defined_by_params(
            np.array([1.0, 2.0, 3.0]), self.inputs_, 2)

        kernel = TasksKernel.define_kernel_from_array(2, np.array([1.0, 2.0, 3.0]))
        assert np.all(result == kernel.cov(np.array(self.inputs_)))

    def test_evaluate_grad_defined_by_params_respect_params(self):
        result = TasksKernel.evaluate_grad_defined_by_params_respect_params(
            np.array([1.0, 2.0, 3.0]), self.inputs_, 2)
        kernel = TasksKernel.define_kernel_from_array(2, np.array([1.0, 2.0, 3.0]))

        grad_kernel = kernel.gradient_respect_parameters(self.inputs_)

        for i in range(2):
            assert np.all(result[i] == grad_kernel['lower_triangular'][i])

    def test_hypers_as_list(self):
        assert self.task_kernel.hypers_as_list == [self.lower_triang]

    def test_hypers_values_as_array(self):
        assert self.task_kernel.hypers_values_as_array == np.array([1.0])

    def test_sample_parameters(self):
        np.random.seed(1)
        value = self.lower_triang.sample_from_prior(2)
        assert np.all(self.task_kernel.sample_parameters(2, 1) == value)

    def test_get_bounds_parameters(self):
        assert [(SMALLEST_NUMBER, LARGEST_NUMBER)] == self.task_kernel.get_bounds_parameters()

    def test_update_value_parameters(self):
        self.task_kernel.update_value_parameters(np.array([2]))

        assert self.task_kernel.lower_triang.value == np.array([2])
        assert self.task_kernel.chol_base_cov_matrix == np.exp(np.array([[2]]))
        npt.assert_almost_equal(self.task_kernel.base_cov_matrix, np.exp(np.array([[4]])))

    def test_define_default_kernel(self):
        kern = TasksKernel.define_default_kernel(1)
        assert kern.lower_triang.value == np.array([0])
        assert kern.name == TASKS_KERNEL_NAME
        assert kern.dimension == 1
        assert kern.dimension_parameters == 1
        assert kern.n_tasks == 1
        assert kern.base_cov_matrix is None
        assert kern.chol_base_cov_matrix is None

        kern_1 = TasksKernel.define_default_kernel(1, None, np.array([3]))
        assert kern_1.lower_triang.value == np.array([3])
        assert kern.name == TASKS_KERNEL_NAME
        assert kern.dimension == 1
        assert kern.dimension_parameters == 1
        assert kern.n_tasks == 1
        assert kern.base_cov_matrix is None
        assert kern.chol_base_cov_matrix is None

    def test_compare_kernels(self):
        kernel = TasksKernel.define_kernel_from_array(1, np.ones(1))

        kernel_ = copy.deepcopy(kernel)
        kernel_.name = 'a'
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension = 2
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension_parameters = 5
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.n_tasks = 5
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.lower_triang.value = np.array([0])
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.base_cov_matrix = np.array([[1]])
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.chol_base_cov_matrix = np.array([[1]])
        assert TasksKernel.compare_kernels(kernel, kernel_) is False

    def test_define_prior_parameters(self):
        data = {
            'points': np.array([[0]]),
            'evaluations': np.array([1]),
            'var_noise': None,
        }

        dimension = 2

        result = TasksKernel.define_prior_parameters(data, dimension)

        assert result == {
            LOWER_TRIANG_NAME: [0.0, -9.2103403719761818, 0.0],
        }
