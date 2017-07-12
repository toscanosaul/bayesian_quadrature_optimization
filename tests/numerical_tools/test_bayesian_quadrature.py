import unittest

import numpy as np
import numpy.testing as npt

from copy import deepcopy

from stratified_bayesian_optimization.models.gp_fitting_gaussian import (
    GPFittingGaussian,
)
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions


class TestBayesianQuadrature(unittest.TestCase):

    def setUp(self):
        self.training_data_complex = {
            "evaluations": [1.0, 1.1],
            "points": [[42.2851784656, 0], [42.3851784656, 0]],
            "var_noise": []}

        self.complex_gp = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            self.training_data_complex, [2, 1, 1])

        self.gp = BayesianQuadrature(self.complex_gp, [0], UNIFORM_FINITE, {TASKS: 1})

        training_data_complex = {
            "evaluations": [1.0, 1.1],
            "points": [[42.2851784656, 0], [42.3851784656, 1]],
            "var_noise": []}

        self.complex_gp_2 = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_complex, [3, 1, 2])

        self.gp_2 = BayesianQuadrature(self.complex_gp_2, [0], UNIFORM_FINITE, {TASKS: 2})


    def test_evaluate_quadrature_cross_cov(self):
        point = np.array([[1.0]])
        points_2 = np.array([[42.2851784656, 0], [42.3851784656, 0]])

        parameters_kernel = self.gp.gp.kernel.hypers_values_as_array
        value = self.gp.evaluate_quadrature_cross_cov(point, points_2, parameters_kernel)

        value_1 = self.gp.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                np.array([[42.2851784656, 0]]), parameters_kernel)
        value_2 = self.gp.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                np.array([[42.3851784656, 0]]), parameters_kernel)
        assert value[0] == value_1[0, 0]
        assert value[1] == value_2[0, 0]

        point = np.array([[1.0]])
        points_2 = np.array([[42.2851784656, 0], [42.3851784656, 1]])

        parameters_kernel = self.gp_2.gp.kernel.hypers_values_as_array
        value = self.gp_2.evaluate_quadrature_cross_cov(point, points_2, parameters_kernel)

        value_1 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                  np.array([[42.2851784656, 0]]), parameters_kernel)
        value_2 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 1.0]]),
                                                  np.array([[42.2851784656, 0]]), parameters_kernel)

        assert value[0] == np.mean([value_1, value_2])

        value_1 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 0.0]]),
                                                  np.array([[42.3851784656, 1]]), parameters_kernel)
        value_2 = self.gp_2.gp.evaluate_cross_cov(np.array([[1.0, 1.0]]),
                                                  np.array([[42.3851784656, 1]]), parameters_kernel)

        assert value[1] == np.mean([value_1, value_2])

    def test_compute_posterior_parameters_kg(self):
        points = np.array([[42.0], [42.1], [41.0]])
        candidate_point = np.array([[41.0, 0]])
        value = self.gp_2.compute_posterior_parameters_kg(points, candidate_point)

        n_samples = 150
        point = np.array([[41.0]])
        samples = self.gp_2.gp.sample_new_observations(candidate_point, n_samples, 1)
        a_n = []
        points_x = deepcopy(self.gp_2.gp.data['points'])
        points_x = np.concatenate((points_x, candidate_point))

        for i in xrange(n_samples):
            evaluations = deepcopy(self.gp_2.gp.data['evaluations'])
            evaluations = np.concatenate((evaluations, [samples[i]]))
            val = self.gp_2.compute_posterior_parameters(point, historical_evaluations=evaluations,
                                                         historical_points=points_x, cache=False)
            a_n.append(val['mean'])

        npt.assert_almost_equal(np.mean(a_n), value['a'][2], decimal=1)
        npt.assert_almost_equal(np.var(a_n),  value['b'][2], decimal=1)

    def test_gradient_posterior_mean(self):
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

        function = function[0, :]

        training_data = {
            'evaluations': list(function),
            'points': points,
            "var_noise": [],
        }

        gaussian_p = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2])
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)

        gp = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})

        point = np.array([[80.5]])

        # Test evaluate_grad_quadrature_cross_cov
        grad = gp.evaluate_grad_quadrature_cross_cov(point, gp.gp.data['points'],
                                                     gp.gp.kernel.hypers_values_as_array)

        dh = 0.00001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point:
            gp.evaluate_quadrature_cross_cov(point, gp.gp.data['points'],
                                             gp.gp.kernel.hypers_values_as_array),
            point, np.array([dh]))

        for i in xrange(grad.shape[1]):
            npt.assert_almost_equal(finite_diff[0][i], grad[0, i], decimal=1)

        npt.assert_almost_equal(finite_diff[0], grad[0, :], decimal=1)

        # Test gradient_posterior_mean
        gradient = gp.gradient_posterior_mean(point)

        dh = 0.0001
        finite_diff = FiniteDifferences.forward_difference(
            lambda points:
            gp.compute_posterior_parameters(points, only_mean=True)['mean'],
            point, np.array([dh]))

        npt.assert_almost_equal(finite_diff[0], gradient[0], decimal=5)

    def test_optimize_posterior_mean(self):
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

        function = function[0, :]

        training_data = {
            'evaluations': list(function),
            'points': points,
            "var_noise": [],
        }

        gaussian_p = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2], bounds_domain=[[0, 100]])
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)

        gp = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})

        random_seed = 10
        sol = gp.optimize_posterior_mean(random_seed=random_seed)

        n_points = 1000
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        evaluations = gp.compute_posterior_parameters(points, only_mean=True)['mean']

        point = points[np.argmax(evaluations), 0]
        index = np.argmax(evaluations)

        assert sol['optimal_value'] >= evaluations[index]
        npt.assert_almost_equal(sol['solution'], point, decimal=2)
