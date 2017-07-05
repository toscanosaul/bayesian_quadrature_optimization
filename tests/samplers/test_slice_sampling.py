import unittest

import numpy as np
import numpy.random as npr
from scipy.stats import norm
import numpy.testing as npt

from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling
from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
)


class TestSliceSampling(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)
        # n_points = 100
        # normal_noise = np.random.normal(0, 1.0, n_points)
        # points = np.linspace(0, 10, n_points)
        # points = points.reshape([n_points, 1])
        # kernel = Matern52.define_kernel_from_array(1, np.array([2.0, 1.0]))
        # function = SampleFunctions.sample_from_gp(points, kernel)
        # function = function[0, :]
        # evaluations = function + normal_noise + 10.0
        # self.training_data_gp = {
        #     "evaluations":list(evaluations),
        #     "points": points,
        #     "var_noise":[]}
        #
        # bounds = [[0, 11.0]]
        # self.gp_gaussian = GPFittingGaussian([MATERN52_NAME], self.training_data_gp, [1], bounds)
        # self.log_prob = self.gp_gaussian.log_prob_parameters
        #
        # self.slice_all_params =  SliceSampling(self.log_prob)
        #
        # def log_prob_2(parameters, a):
        #     vect = np.array([parameters[0], parameters[1], a , parameters[2]])
        #     return self.log_prob(vect)
        #
        # self.slice = SliceSampling(log_prob_2, **{'component_wise': False})
        #
        # def log_prob_3(parameters, a, b, c):
        #     vect = np.array([a, b, parameters[0] , c])
        #     return self.log_prob(vect)
        #
        # self.slice_ = SliceSampling(log_prob_3)
        #
        # self.norm_density = lambda x: norm.pdf(x)[0]
        #
        # np.random.seed(1)
        # data = np.random.normal(0, 1, 30)
        #
        # def logp(params):
        #     std = params[1]
        #     mu = params[0]
        #     if std <= 0:
        #         return -np.inf
        #     return np.sum(np.log(norm.pdf(data, mu, std)))

       # self.slice_2 = SliceSampling(logp)

    def test_slice_sample(self):
        np.random.seed(1)
        samples = []
        point = np.array([5.0, 100.0])
      #
      #   for j in range(200):
      #       point = self.slice_2.slice_sample(point)
      #       samples.append(point)
      #   z = samples[10::5]
      #   npt.assert_almost_equal(np.mean([x[0] for x in z]), 0.0, decimal=1)
      #   npt.assert_almost_equal(2.0 * np.std([x[0] for x in z]) /len(z), 0.0, decimal=2)
      #   npt.assert_almost_equal(np.mean([x[1] for x in z]), 1.0, decimal=1)
      #   npt.assert_almost_equal(2.0 * np.std([x[1] for x in z]) /len(z), 0.0, decimal=2)
      #
      #   samples = []
      #   point = np.array([0.1, 0.7, 0.2])
      #   point_ = np.array([0.8])
      #  # point = np.array([0.1, 0.7, 0.8, 0.5])
      #   np.random.seed(1)
      #   for j in range(2):
      #     #  point = self.slice_all_params.slice_sample(point)
      #     #  print point
      #     #  samples.append(point)
      #       point = self.slice.slice_sample(point, *point_)
      #       point_ = self.slice_.slice_sample(point_, *point)
      #       sample_p = np.zeros(4)
      #       sample_p[[0, 1, 3]] = point
      #       sample_p[2] = point_
      #       samples.append(sample_p)
      #      # print sample_p
      #  # z = samples[300::5]
      #  # print np.mean([x[0] for x in z])
      #  # print np.mean([x[1] for x in z])
      #  # print np.mean([x[2] for x in z])
      #  # print np.mean([x[3] for x in z])
      #
      #
      # #  print np.mean([x[3] for x in z])
      #   npt.assert_almost_equal(np.mean([x[0] for x in z]), 1.0, decimal=0)
      #   npt.assert_almost_equal(2.0 * np.std([x[0] for x in z]) /len(z), 0.0, decimal=1)
      #   npt.assert_almost_equal(np.mean([x[1] for x in z]), 50.0, decimal=-1)
      #   npt.assert_almost_equal(2.0 * np.std([x[1] for x in z]) /len(z), 0.0, decimal=0)
      #   npt.assert_almost_equal(np.mean([x[2] for x in z]), 200.0, decimal=0)
      #   npt.assert_almost_equal(2.0 * np.std([x[2] for x in z]) /len(z), 0.0, decimal=0)

        # TODO: WRITE self.slice_all_params

    def test_find_x_interval(self):
        np.random.seed(1)
        # upper = self.slice.sigma * npr.rand()
        # lower = upper - self.slice.sigma
        #
        # point = np.array([0.25, 0.0, 100.0, 1.0])
        # dimensions = len(point)
        # direction = np.zeros(dimensions)
        # direction[0] = 1.0
       # llh = np.log(npr.rand()) + self.slice.directional_log_prob(0.0, direction, point)
       # upper, lower = self.find_x_interval(llh, lower, upper, direction, point)
        #print upper, lower
        # self.slice.find_x_interval()
