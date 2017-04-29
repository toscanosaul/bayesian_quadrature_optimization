from __future__ import absolute_import

import unittest

from doubles import expect

import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel, GradientTasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity


class TestTasksKernel(unittest.TestCase):

    def setUp(self):
        self.n_tasks = 1
        self.lower_triang = ParameterEntity('lower_triang', np.array([1.0]), None)
        self.task_kernel = TasksKernel(self.n_tasks, self.lower_triang)
        self.inputs = np.array([[0]])

    def test_hypers(self):
        assert {'lower_triang': self.lower_triang} == self.task_kernel.hypers

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
        gradient[0] = np.array([[0]])

        assert self.task_kernel.gradient_respect_parameters(self.inputs) == gradient

    def test_grad_respect_point(self):
        assert self.task_kernel.grad_respect_point(self.inputs, self.inputs) == np.array([[0]])

    def test_gradient_respect_parameters_gradient_class(self):
        grad = GradientTasksKernel.gradient_respect_parameters(np.array([[np.exp(1.0)]]), 1)
        assert len(grad) == 1
        npt.assert_almost_equal(grad[0], np.array([[2.0 * np.exp(2.0)]]))
