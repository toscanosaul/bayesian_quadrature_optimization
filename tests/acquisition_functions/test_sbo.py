import unittest

from mock import create_autospec
import mock

from doubles import expect

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


def simple_affine_break_points(a, b):
    return [1], 2

class TestSBO(unittest.TestCase):
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

        self.gp = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})


        # ver lo de domain entity
        self.bounds_domain_x = BoundsEntity({
            'lower_bound': 0,
            'upper_bound': 100,
        })

        self.spec = {
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [self.bounds_domain_x],
            'number_points_each_dimension': [100],
            'problem_name': 'a',
        }

        domain = DomainService.from_dict(self.spec)

        self.kernel = None
        self.domain = create_autospec(DomainEntity)

        self.sbo = SBO(self.gp, np.array(domain.discretization_domain_x))


        training_data_simple = {
            'evaluations': list(function[0:1]),
            'points': points[0:1, :],
            "var_noise": [],
        }
        gaussian_p_simple = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_simple, [2, 1, 2], bounds_domain=[[0, 100], [0, 1]], type_bounds=[0, 1])
        gaussian_p_simple.update_value_parameters(self.sbo.bq.gp.get_value_parameters_model)
        gp_simple = BayesianQuadrature(gaussian_p_simple, [0], UNIFORM_FINITE, {TASKS: 2})

        self.sbo_simple = SBO(gp_simple, np.array([[2]]))


        training_data_med = {
            'evaluations': list(function[0:5]),
            'points': points[0:5, :],
            "var_noise": [],
        }
        gaussian_p_med = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_med, [2, 1, 2], bounds_domain=[[0, 100], [0, 1]], type_bounds=[0, 1])
        gaussian_p_med.update_value_parameters(self.sbo.bq.gp.get_value_parameters_model)
        gp_med = BayesianQuadrature(gaussian_p_med, [0], UNIFORM_FINITE, {TASKS: 2})

        self.sbo_med = SBO(gp_med, np.array(domain.discretization_domain_x))


    def test_evaluate(self):
        point = np.array([[52.5, 0]])
        value = self.sbo.evaluate(point)
        discretization = self.sbo.discretization


        bq = self.sbo.bq
        gp = self.sbo.bq.gp
        random_seed = 1
        n_samples = 40
        samples = gp.sample_new_observations(point, n_samples, random_seed)
        np.random.seed(random_seed)
        points_x = deepcopy(gp.data['points'])
        points_x = np.concatenate((points_x, point))

        posterior_means = bq.compute_posterior_parameters(discretization, only_mean=True,
                                                          cache=False)['mean']

        max_mean = posterior_means[np.argmax(posterior_means)]
        max_values = []

        for i in xrange(n_samples):
            evaluations = deepcopy(gp.data['evaluations'])
            evaluations = np.concatenate((evaluations, [samples[i]]))
            val = bq.compute_posterior_parameters(discretization,
                                                  historical_evaluations=evaluations,
                                                  historical_points=points_x, cache=False,
                                                  only_mean=True)
            values = val['mean']
            max_values.append(values[np.argmax(values)])

        kg = np.mean(max_values) - max_mean
        std = np.std(max_values) / n_samples
        assert kg - 1.96 * std <= value <= kg + 1.96 * std

        point = self.points[0:1, :]
        assert self.sbo.evaluate(point) == 0

    def test_evaluate_gradient(self):
        candidate = np.array([[52.5, 0]])
        self.sbo.clean_cache()
        grad = self.sbo.evaluate_gradient(candidate)

        dh = 0.1
        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.sbo.evaluate(point.reshape((1, len(point)))),
            np.array([52.5, 0]), np.array([dh]))
        npt.assert_almost_equal(finite_diff[1], grad[1], decimal=5)
        npt.assert_almost_equal(finite_diff[0], grad[0], decimal=2)

        point = self.points[0:1, :]
        assert np.all(self.sbo.evaluate_gradient(point) == [0, 0])

    def test_evaluate_gradient_keeping_one_point(self):
        candidate = np.array([[52.5, 0]])

        grad = self.sbo_simple.evaluate_gradient(candidate)
        assert np.all(grad == np.array([0, 0]))

        dh = 0.1
        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.sbo_simple.evaluate(point.reshape((1, len(point)))),
            np.array([52.5, 0]), np.array([dh]))
        npt.assert_almost_equal(finite_diff[1], grad[1])
        npt.assert_almost_equal(finite_diff[0], grad[0])

    def test_hvoi(self):
        b = np.array([1, 2])
        c = np.array([3, 4])
        keep = [0]
        z = self.sbo.hvoi(b, c, keep)
        assert z == 0

    def test_optimization(self):
        val = self.sbo_med.optimize(random_seed=1, parallel=False)
        # Benchmark numbers obtained after optimizing the function manually, i.e. plot the function
        # and find the maximum.
        npt.assert_almost_equal(2018.8827643498898, val['optimal_value'], decimal=3)
        npt.assert_almost_equal([ 99.98636451, 0], val['solution'], decimal=4)

        self.sbo_med.opt_separing_domain = False
        val = self.sbo_med.optimize(random_seed=1)
        npt.assert_almost_equal([99.98636451, 0], val['solution'], decimal=4)
        npt.assert_almost_equal(2018.8827643498898, val['optimal_value'], decimal=3)

    def test_optimization_error(self):
        expect(Parallel).run_function_different_arguments_parallel.and_return({0: None})
        with self.assertRaises(Exception):
            self.sbo_med.optimize(random_seed=1, parallel=False)
