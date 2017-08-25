import unittest

from mock import create_autospec
import mock

from doubles import expect

import warnings

import numpy as np
import numpy.testing as npt

from copy import deepcopy

from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.entities.domain import DomainEntity
from stratified_bayesian_optimization.lib.constant import (
    PRODUCT_KERNELS_SEPARABLE,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.entities.domain import(
    BoundsEntity,
    DomainEntity,
)
from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.lib.affine_break_points import (
    AffineBreakPoints,
)
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.lib.util import (
    wrapper_objective_acquisition_function,
    wrapper_gradient_acquisition_function,
)


class TestEI(unittest.TestCase):
    def setUp(self):

        np.random.seed(5)
        n_points = 100
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        tasks = np.random.randint(2, size=(n_points, 1))

        add = [10, -10]
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)

        for i in xrange(n_points):
            function[0, i] += add[tasks[i, 0]]
        points = np.concatenate((points, tasks), axis=1)
        self.points = points
        self.evaluations = function[0, :]

        function = function[0, :]

        training_data = {
            'evaluations': list(function),
            'points': points,
            "var_noise": [],
        }

        gaussian_p = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2], bounds_domain=[[0, 100], [0, 1]], type_bounds=[0, 1])
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)

        self.gp = gaussian_p

        self.bq = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2},
                                     model_only_x=True)

        self.ei = EI(self.gp)

        self.ei_2 = EI(self.bq)


    def test_evaluate(self):
        point = np.array([[97.5, 0]])
        val = self.ei.evaluate(point)

        maximum = np.max(self.evaluations)

        n_samples = 10000
        samples = self.gp.sample_new_observations(point, n_samples, random_seed=1)

        evals = np.clip(samples - maximum, 0, None)

        npt.assert_almost_equal(val, np.mean(evals), decimal=2)

    def test_evaluate_bq(self):
        point =  np.array([[97.5]])
        val = self.ei_2.evaluate(point)

        maximum = 0.831339057477
        n_samples = 10000
        samples = self.bq.sample_new_observations(point, n_samples, random_seed=1)

        evals = np.clip(samples - maximum, 0, None)
        npt.assert_almost_equal(val, np.mean(evals), decimal=2)

    def test_evaluate_gradient(self):
        point = np.array([[91.5, 0]])
        grad = self.ei.evaluate_gradient(point)

        dh = 0.0001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.ei.evaluate(point.reshape((1, len(point)))),
            np.array([91.5, 0]), np.array([dh]))

        npt.assert_almost_equal(finite_diff[0], grad[0], decimal=2)
        npt.assert_almost_equal(finite_diff[1], grad[1], decimal=3)

    def test_evaluate_gradient_bq(self):
        point =  np.array([[91.5]])
        grad = self.ei_2.evaluate_gradient(point)

        dh = 0.0001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.ei_2.evaluate(point.reshape((1, len(point)))),
            np.array([91.5]), np.array([dh]))

        npt.assert_almost_equal(finite_diff[0], grad[0], decimal=2)


    def test_optimize(self):
        np.random.seed(2)
        opt = self.ei.optimize(random_seed=1, n_restarts=120)

        evaluations = self.ei.generate_evaluations('1', '2', '3', 1, 1, 1, [100], 2)
        npt.assert_almost_equal(opt['optimal_value'], np.max(evaluations))

    def test_optimize_bq(self):
        np.random.seed(2)
        opt = self.ei_2.optimize(random_seed=1, n_restarts=50)

        evaluations = self.ei_2.generate_evaluations('1', '2', '3', 1, 1, 1, [100], 0)

        npt.assert_almost_equal(opt['optimal_value'], np.max(evaluations))

    def test_evaluate_parameters(self):
        point = np.array([[97.5, 0]])

        np.random.seed(1)
        val_2 = self.ei.evaluate(point, 1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1]))

        val = self.ei.evaluate(point)

        self.gp.var_noise.value[0] = 1.0
        self.gp.mean.value[0] = 5.0
        self.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        val_1 = self.ei.evaluate(point)

        npt.assert_almost_equal(val_1, val_2)

    def test_evaluate_bq_parameters(self):
        point =  np.array([[97.5]])

        np.random.seed(1)
        val_2 = self.ei_2.evaluate(point, 1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1]))
        val = self.ei_2.evaluate(point)

        self.bq.gp.var_noise.value[0] = 1.0
        self.bq.gp.mean.value[0] = 5.0
        self.bq.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        val_1 = self.ei_2.evaluate(point)

        npt.assert_almost_equal(val_1, val_2)

    def test_gradient_parameters(self):
        point = np.array([[91.5, 0]])

        np.random.seed(1)
        grad_2 = self.ei.evaluate_gradient(point, *(1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1])))
        grad = self.ei.evaluate_gradient(point)

        self.gp.var_noise.value[0] = 1.0
        self.gp.mean.value[0] = 5.0
        self.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        grad_1 = self.ei.evaluate_gradient(point)

        npt.assert_almost_equal(grad_1, grad_2)

    def test_gradient_bq_parameters(self):
        point =  np.array([[91.5]])
        np.random.seed(1)
        grad_2 = self.ei_2.evaluate_gradient(point, *(1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1])))

        grad = self.ei_2.evaluate_gradient(point)
        self.bq.gp.var_noise.value[0] = 1.0
        self.bq.gp.mean.value[0] = 5.0
        self.bq.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        grad_1 = self.ei_2.evaluate_gradient(point)

        npt.assert_almost_equal(grad_2, grad_1)

    def test_combine_ei_gradient(self):
        point = np.array([[91.5, 0]])

        np.random.seed(1)

        val = self.ei.evaluate(point, *(1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1])))


        grad =  self.ei.evaluate_gradient(point, *(1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1])))

        self.ei.clean_cache()
        np.random.seed(1)
        grad_1 =  self.ei.evaluate_gradient(point, *(1.0, 5.0, np.array([50.0, 8.6, -3.0, -0.1])))

        npt.assert_almost_equal(grad, grad_1)

    def test_evaluate_ei_sample_parameters(self):
        point = np.array([91.5, 0])
        self.ei.gp.thinning = 5
        self.ei.gp.n_burning = 100

        n_samples_parameters = 15
        np.random.seed(1)
        self.gp.start_new_chain()
        self.gp.sample_parameters(n_samples_parameters)
        value = wrapper_objective_acquisition_function(point, self.ei, n_samples_parameters)


        gradient = wrapper_gradient_acquisition_function(point, self.ei, n_samples_parameters)
        np.random.seed(1)
        sol = self.ei.optimize(None, 1, True, 10,
                                    n_samples_parameters=n_samples_parameters)
        npt.assert_almost_equal(value, 0.297100121625)
        npt.assert_almost_equal(gradient, np.array([0.00058253, 0]))
        npt.assert_almost_equal(sol['optimal_value'], 0.30205775381169897)

    def test_optimize_ei(self):
        np.random.seed(2)
        opt = self.ei.optimize(random_seed=1, n_restarts=120)

        np.random.seed(2)
        opt_2 = self.ei.optimize(random_seed=1, n_restarts=120, n_best_restarts=10)


        npt.assert_almost_equal(opt['optimal_value'], opt_2['optimal_value'])

    def test_optimize_ei_2(self):
        self.ei.gp.thinning = 5
        self.ei.gp.n_burning = 100
        n_samples_parameters = 15

        np.random.seed(2)
        opt = self.ei.optimize(random_seed=1, n_restarts=15, start_new_chain=True,
                               n_samples_parameters=n_samples_parameters)

        np.random.seed(2)
        opt_2 = self.ei.optimize(random_seed=1, n_restarts=100, n_best_restarts=10,
                                 start_new_chain=True, n_samples_parameters=n_samples_parameters)
        assert opt_2['optimal_value'] >= opt['optimal_value']
