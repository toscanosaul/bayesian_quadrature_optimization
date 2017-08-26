import unittest

import numpy as np
import numpy.testing as npt

from mock import patch, mock_open

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
    QUADRATURES,
    POSTERIOR_MEAN,
    B_NEW,
    DOGLEG,
)
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.util.json_file import JSONFile


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

        np.random.seed(5)
        n_points = 100
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        tasks = np.random.randint(2, size=(n_points, 1))

        add = [10, -10]
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)
        self.original_function = function

        self.max_value = function[0, np.argmax(function)]
        self.max_point = points[np.argmax(function), 0]
        for i in xrange(n_points):
            function[0, i] += add[tasks[i, 0]]
        points = np.concatenate((points, tasks), axis=1)

        function = function[0, :]

        training_data = {
            'evaluations': list(function),
            'points': points,
            "var_noise": [],
        }

        training_data_2 = {
            'evaluations': list(function[[0, 30, 50, 90, 99]]),
            'points': points[[0, 30, 50, 90, 99], :],
            "var_noise": [],
        }

        gaussian_p = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data, [2, 1, 2], bounds_domain=[[0, 100]])
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)

        gaussian_p_2 = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            training_data_2, [2, 1, 2], bounds_domain=[[0, 100]])
        gaussian_p_2 = gaussian_p.fit_gp_regression(random_seed=1314938)

        self.gp_complete_2 = BayesianQuadrature(gaussian_p_2, [0], UNIFORM_FINITE, {TASKS: 2})
        self.gp_complete = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})


    def test_constructor(self):
        gp = BayesianQuadrature(self.complex_gp, [0], UNIFORM_FINITE, {})
        assert gp.parameters_distribution == {TASKS: 1}

    def test_get_cached_data(self):
        gp = BayesianQuadrature(self.complex_gp, [0], UNIFORM_FINITE, {})
        gp.cache_quadratures['a'] = 1
        gp.cache_posterior_mean['a'] = 2
        gp.cache_quadrature_with_candidate['b'] = 3

        assert gp._get_cached_data('a', QUADRATURES) == 1
        assert gp._get_cached_data('a', POSTERIOR_MEAN) == 2
        assert gp._get_cached_data('b', B_NEW) == 3

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
        npt.assert_almost_equal(np.var(a_n),  (value['b'][2]) ** 2, decimal=1)

    def test_gradient_posterior_mean(self):
        gp = self.gp_complete

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
        gp = self.gp_complete

        random_seed = 10
        sol = gp.optimize_posterior_mean(random_seed=random_seed)

        n_points = 1000
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        evaluations = gp.compute_posterior_parameters(points, only_mean=True)['mean']

        point = points[np.argmax(evaluations), 0]
        index = np.argmax(evaluations)

        npt.assert_almost_equal(sol['optimal_value'][0], evaluations[index])
        npt.assert_almost_equal(sol['solution'], point, decimal=1)

        bounds_x = [gp.gp.bounds[i] for i in xrange(len(gp.gp.bounds)) if i in
                    gp.x_domain]
        random_seed = 10
        np.random.seed(10)
        start = DomainService.get_points_domain(1, bounds_x,
                                                type_bounds=len(gp.x_domain) * [0])
        start = np.array(start[0])

        var_noise = gp.gp.var_noise.value[0]
        parameters_kernel = gp.gp.kernel.hypers_values_as_array
        mean = gp.gp.mean.value[0]

        index_cache = (var_noise, mean, tuple(parameters_kernel))
        if index_cache not in gp.optimal_solutions:
            gp.optimal_solutions[index_cache] = []
        gp.optimal_solutions[index_cache].append({'solution': start})

        sol_2 = gp.optimize_posterior_mean(random_seed=random_seed)
        npt.assert_almost_equal(sol_2['optimal_value'], sol['optimal_value'])
        npt.assert_almost_equal(sol['solution'], sol_2['solution'], decimal=3)


    def test_optimize_posterior_mean_hessian(self):
        gp = self.gp_complete
        random_seed = 1
        sol_3 = gp.optimize_posterior_mean(random_seed=random_seed, method_opt=DOGLEG)

        gp.clean_cache()
        sol_2 = gp.optimize_posterior_mean(random_seed=random_seed)
        assert sol_3['solution'] == sol_2['solution']
        npt.assert_almost_equal(sol_3['optimal_value'], sol_2['optimal_value'], decimal=2)


    def test_evaluate_grad_quadrature_cross_cov_resp_candidate(self):
        candidate_point = np.array([[51.5, 0]])
        points = np.array([[51.3], [30.5], [95.1]])
        parameters = self.gp_complete.gp.kernel.hypers_values_as_array
        sol = self.gp_complete.evaluate_grad_quadrature_cross_cov_resp_candidate(
            candidate_point, points, parameters
        )

        gp = self.gp_complete

        dh = 0.000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point:
            gp.evaluate_quadrature_cross_cov(points[0:1, :], point.reshape((1, len(point))),
                                             parameters),
            candidate_point[0, :], np.array([dh]))
        npt.assert_almost_equal(sol[0, 0], finite_diff[0][0], decimal=2)
        assert sol[1, 0] == finite_diff[1]

        dh = 0.000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point:
            gp.evaluate_quadrature_cross_cov(points[1:2, :], point.reshape((1, len(point))),
                                             parameters),
            candidate_point[0, :], np.array([dh]))
        npt.assert_almost_equal(sol[0, 1], finite_diff[0][0], decimal=1)
        assert sol[1, 1] == finite_diff[1]

        dh = 0.000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda point:
            gp.evaluate_quadrature_cross_cov(points[2:3, :], point.reshape((1, len(point))),
                                             parameters),
            candidate_point[0, :], np.array([dh]))
        npt.assert_almost_equal(sol[0, 2], finite_diff[0][0], decimal=2)
        assert sol[1, 2] == finite_diff[1]

    def test_gradient_vector_b(self):
        np.random.seed(5)
        n_points = 10
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
      #  gp = self.gp_complete
        candidate_point = np.array([[84.0, 1]])
        points = np.array([[99.5], [12.1], [70.2]])
        value = gp.gradient_vector_b(candidate_point, points, cache=False)

        dh_ = 0.0000001
        dh = [dh_]
        finite_diff = FiniteDifferences.forward_difference(
            lambda point:
            gp.compute_posterior_parameters_kg(
                points, point.reshape((1, len(point))), cache=False)['b'],
            candidate_point[0, :], np.array(dh))
        npt.assert_almost_equal(finite_diff[0], value[:, 0], decimal=5)
        assert np.all(finite_diff[1] == value[:, 1])

        value_2 = gp.gradient_vector_b(candidate_point, points, cache=True)
        assert np.all(value_2 == value)

    def test_sample_new_observations(self):
        gp = self.gp_complete
        point = np.array([[5.1]])
        samples = gp.sample_new_observations(point, 2, random_seed=1)
        assert len(samples) == 2

    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_write_debug_data(self, mock_mkdir, mock_exists):
        mock_exists.return_value = False
        with patch('__builtin__.open', mock_open()):
            self.gp.write_debug_data("a", "b", "c", "d", "e")
            JSONFile.write([], "a")
        mock_mkdir.assert_called_with('data/debugging/a')

    def test_evaluate_posterior_mean_params(self):
        point = np.array([[97.5]])

        np.random.seed(1)
        val_2 = self.gp_complete.objective_posterior_mean(point[0, :], 1.0, 5.0,
                                                 np.array([50.0, 8.6, -3.0, -0.1]))

        val = self.gp_complete.objective_posterior_mean(point[0, :])

        self.gp_complete.gp.var_noise.value[0] = 1.0
        self.gp_complete.gp.mean.value[0] = 5.0
        self.gp_complete.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        val_1 = self.gp_complete.objective_posterior_mean(point[0, :])

        npt.assert_almost_equal(val_1, val_2)

    def test_evaluate_grad_posterior_mean_params(self):
        point = np.array([[97.5]])

        np.random.seed(1)
        val_2 = self.gp_complete.grad_posterior_mean(point[0, :], 1.0, 5.0,
                                                 np.array([50.0, 8.6, -3.0, -0.1]))

        val = self.gp_complete.grad_posterior_mean(point[0, :])

        self.gp_complete.gp.var_noise.value[0] = 1.0
        self.gp_complete.gp.mean.value[0] = 5.0
        self.gp_complete.gp.kernel.update_value_parameters(np.array([50.0, 8.6, -3.0, -0.1]))

        np.random.seed(1)
        val_1 = self.gp_complete.grad_posterior_mean(point[0, :])

        npt.assert_almost_equal(val_1, val_2)

    def test_optimize_posterior_mean_samples(self):
        np.random.seed(5)
        n_points = 100
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        tasks = np.random.randint(2, size=(n_points, 1))

        add = [10, -10]
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)
        max_value = function[0, np.argmax(function)]
        max_point = points[np.argmax(function), 0]

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
            training_data, [2, 1, 2], bounds_domain=[[0, 100]], max_steps_out=1000)
        gaussian_p = gaussian_p.fit_gp_regression(random_seed=1314938)
        gp = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE, {TASKS: 2})


        random_seed = 10

        n_samples_parameters = 15
        gp.gp.thinning = 10
        gp.gp.n_burning = 500

        sol_2 = gp.optimize_posterior_mean(random_seed=random_seed, n_best_restarts=10,
                                           n_samples_parameters=n_samples_parameters,
                                           start_new_chain=True)

        assert max_point == sol_2['solution']
        npt.assert_almost_equal(max_value, sol_2['optimal_value'], decimal=3)

    def test_compute_hessian_parameters_for_sample(self):
        point = np.array([[95.0]])
        candidate_point = np.array([[99.15, 0]])
        val = self.gp_complete_2.compute_hessian_parameters_for_sample(point, candidate_point)

        dh = 0.01
        finite_diff = FiniteDifferences.second_order_central(
            lambda x: self.gp_complete_2.compute_parameters_for_sample(
                x.reshape((1, len(point))), candidate_point, clear_cache=False)['a'],
            point[0, :], np.array([dh])
        )

        npt.assert_almost_equal(finite_diff[(0, 0)], val['a'][0,:], decimal=5)

        dh = 0.1
        finite_diff = FiniteDifferences.second_order_central(
            lambda x: self.gp_complete_2.compute_parameters_for_sample(
                x.reshape((1, len(point))), candidate_point, clear_cache=False)['b'],
            point[0, :], np.array([dh])
        )


        npt.assert_almost_equal(finite_diff[(0, 0)], val['b'], decimal=5)

    def test_hessian_posterior_mean(self):

        gp = self.gp_complete

        point = np.array([[80.5]])

        # Test evaluate_grad_quadrature_cross_cov
        hessian = gp.hessian_posterior_mean(point)

        dh = 0.1
        finite_diff = FiniteDifferences.second_order_central(
            lambda points:
            gp.compute_posterior_parameters(points.reshape((1, len(points))),
                                            only_mean=True)['mean'],
            point[0, :], np.array([dh]))

        npt.assert_almost_equal(finite_diff[(0, 0)], hessian[0, 0])

