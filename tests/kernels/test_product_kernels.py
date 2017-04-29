import unittest

from mock import create_autospec
from doubles import expect

import numpy as np

from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity


class TestProductKernels(unittest.TestCase):

    def setUp(self):
        self.length_scale = ParameterEntity('scale', np.array([1.0]), None)
        self.sigma2 = ParameterEntity('sigma2', np.array([3]), None)
        self.matern52 = Matern52(1, self.length_scale, self.sigma2)

        self.n_tasks = 1
        self.lower_triang = ParameterEntity('lower_triang', np.array([1.0]), None)
        self.task_kernel = TasksKernel(self.n_tasks, self.lower_triang)

        self.tasks = create_autospec(TasksKernel)
        self.kernel = ProductKernels(self.matern52, self.task_kernel)

        self.parameters = {'Matern52': {
            'length_scale': ParameterEntity('scale', np.array([5.0]), None), 'sigma2': self.sigma2},
            'Tasks_Kernel': {
                'lower_triang': ParameterEntity('lower_triang', np.array([10.0]), None)
            }
        }

        self.inputs = {'Matern52': np.array([[5.0]]), 'Tasks_Kernel': np.array([[0]])}

    def test_hypers(self):
        assert self.kernel.hypers == {'Matern52': {
            'scale': self.length_scale, 'sigma2': self.sigma2},
            'Tasks_Kernel': {
                'lower_triang': self.lower_triang
            }
        }

    def test_set_parameters(self):
        self.kernel.set_parameters(self.parameters)

        assert self.kernel.hypers == {'Matern52': {
            'scale': self.parameters['Matern52']['length_scale'], 'sigma2': self.sigma2},
            'Tasks_Kernel': {
                'lower_triang': self.parameters['Tasks_Kernel']['lower_triang']
            }
        }

    def test_cov(self):
        expect(self.kernel).cross_cov.once().and_return(0)
        assert self.kernel.cov(self.inputs) == 0

    def test_cross_cov(self):
        expect(self.matern52).cross_cov.once().and_return(5)
        expect(self.task_kernel).cross_cov.once().and_return(10)

        assert self.kernel.cross_cov(self.inputs, self.inputs) == 50

    def test_gradient_respect_parameters(self):
        expect(self.matern52).cov.once().and_return(5)
        expect(self.task_kernel).cov.once().and_return(10)

        expect(self.matern52).gradient_respect_parameters.once().and_return({
            'scale': 2.0,
            'sigma2': 3.0
        })
        expect(self.task_kernel).gradient_respect_parameters.once().and_return({
            'lower_triang': -1.0
        })

        assert self.kernel.gradient_respect_parameters(self.inputs) == {'Matern52': {
            'scale': 20.0, 'sigma2': 30.0
        }, 'Tasks_Kernel': {'lower_triang': -5.0}}

    def test_grad_respect_point(self):
        expect(self.matern52).cross_cov.once().and_return(5)
        expect(self.task_kernel).cross_cov.once().and_return(10)

        expect(self.matern52).grad_respect_point.once().and_return(-1)
        expect(self.task_kernel).grad_respect_point.once().and_return(3)

        assert self.kernel.grad_respect_point(self.inputs, self.inputs) == \
               {'Matern52': -10, 'Tasks_Kernel': 15}
