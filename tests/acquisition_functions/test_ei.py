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

        self.bq = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})

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
        npt.assert_almost_equal(finite_diff[1], grad[1])

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
        opt = self.ei.optimize(random_seed=1, n_restarts=90)

        evaluations = self.ei.generate_evaluations('1', '2', '3', 1, 1, 1, [100], 2)
        npt.assert_almost_equal(opt['optimal_value'], np.max(evaluations))

    def test_optimize_bq(self):
        np.random.seed(2)
        opt = self.ei_2.optimize(random_seed=1, n_restarts=50)

        evaluations = self.ei_2.generate_evaluations('1', '2', '3', 1, 1, 1, [100], 0)

        npt.assert_almost_equal(opt['optimal_value'], np.max(evaluations))
