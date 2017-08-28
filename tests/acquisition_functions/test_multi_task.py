import unittest

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
from stratified_bayesian_optimization.acquisition_functions.multi_task import MultiTasks


class TestMultiTask(unittest.TestCase):

    def setUp(self):
        np.random.seed(5)
        n_points = 100
        points = np.linspace(0, 100, n_points)
        points = points.reshape([n_points, 1])
        tasks = np.random.randint(2, size=(n_points, 1))

        add = [-10, 10]
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)
        self.function = function

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

        quadrature = BayesianQuadrature(gaussian_p, [0], UNIFORM_FINITE,
                                        parameters_distribution={TASKS: 2},
                                        model_only_x=True)
        self.mt = MultiTasks(quadrature, quadrature.parameters_distribution.get(TASKS))



    def test_optimize(self):
        sol = self.mt.optimize_first(random_seed=1)

        final_sol = self.mt.optimize(random_seed=1)

        assert np.all(final_sol['solution'] == np.array([100, 0]))

    def test_optimize_samples(self):
        self.mt.bq.gp.thinning = 5
        self.mt.bq.gp.n_burning = 100
        self.mt.bq.gp.max_steps_out = 1000
        np.random.seed(1)
        sol = self.mt.optimize(random_seed=1, n_samples_parameters=10, n_restarts=100,
                               n_best_restarts=10)
        npt.assert_almost_equal(sol['solution'], np.array([96.0098558, 1]))

