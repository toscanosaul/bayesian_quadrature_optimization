import unittest

import numpy as np
import numpy.testing as npt
from doubles import expect

from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
)


class TestSliceSampling(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)
        n_points = 100
        normal_noise = np.random.normal(0, 1.0, n_points)
        points = np.linspace(0, 10, n_points)
        points = points.reshape([n_points, 1])
        kernel = Matern52.define_kernel_from_array(1, np.array([2.0, 1.0]))
        function = SampleFunctions.sample_from_gp(points, kernel)
        function = function[0, :]
        evaluations = function + normal_noise + 10.0
        self.training_data_gp = {
            "evaluations":list(evaluations),
            "points": points,
            "var_noise":[]}
        bounds = None
        self.gp_gaussian = GPFittingGaussian([MATERN52_NAME], self.training_data_gp, [1], bounds,
                                             max_steps_out=1000)


    def test_slice_sample(self):
        # Benchmark numbers from Ryan's code.

        np.random.seed(1)
        point = np.array([0.1, 0.7, 0.8, 0.2])

        new_point = self.gp_gaussian.sample_parameters(1, point, 1)[0]
        benchmark_point = np.array([0.17721380376549206, 0.67091995290377726, 2.23209165,
                                             0.17489317792506012])

        npt.assert_almost_equal(new_point, benchmark_point)

        np.random.seed(1)
        point = np.array([0.1, 0.7, 0.8, 0.2])

        sampler = self.gp_gaussian.slice_samplers[0]
        sampler.doubling_step = False

        new_point = self.gp_gaussian.sample_parameters(1, point, 1)[0]
        npt.assert_almost_equal(new_point, benchmark_point)


    def test_acceptable(self):
        sampler = self.gp_gaussian.slice_samplers[0]
        accept = sampler.acceptable(0.75, 1000000, 0, 1.5, np.array([1.0, 0, 0]),
                                    np.array([0.1, 0.7, 0.2]), np.array([0.8]),
                                    *(self.gp_gaussian,))
        assert accept is False

    def test_find_x_interval(self):
        sampler = self.gp_gaussian.slice_samplers[0]
        sampler.doubling_step = False
        interval = sampler.find_x_interval(-2000, 0, 1.5, np.array([1.0, 0, 0]),
                                           np.array([0.1, 0.7, 0.2]), np.array([0.8]),
                                           *(self.gp_gaussian,))
        assert interval == (1001.5, -1.0)

    def test_find_sample(self):
        sampler = self.gp_gaussian.slice_samplers[0]

        with self.assertRaises(Exception):
            sampler.find_sample(0, 1.5, -1000, np.array([1.0, 0, 0]), np.array([-1.0, 0.7, 0.2]),
                                np.array([0.8]), *(self.gp_gaussian,))

        expect(sampler).directional_log_prob.and_return(np.nan)
        with self.assertRaises(Exception):
            sampler.find_sample(0, 1.5, -1000, np.array([1.0, 0, 0]), np.array([-1.0, 0.7, 0.2]),
                                np.array([0.8]), *(self.gp_gaussian,))
