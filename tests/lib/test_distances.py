from __future__ import absolute_import

import unittest

from doubles import expect

import numpy as np

from stratified_bayesian_optimization.lib.distances import Distances

class TestDistances(unittest.TestCase):

    def setUp(self):
        self.ls = np.array([2.0])
        self.x1 = np.array([[3.0]])

    def test_dist_square_length_scale(self):
        assert Distances.dist_square_length_scale(self.ls, self.x1) == [[0.0]]

    def test_gradient_distance_length_scale_respect_ls(self):
        expect(Distances).dist_square_length_scale.once().and_return(np.array([[1.0]]))

        assert Distances.gradient_distance_length_scale_respect_ls(self.ls, self.x1) == \
               {0: np.array([[0.0]])}

    def test_gradient_distance_length_scale_respect_point(self):
        expect(Distances).dist_square_length_scale.once().and_return(np.array([[1.0]]))
        assert Distances.gradient_distance_length_scale_respect_point(self.ls, self.x1, self.x1) \
               == np.array([[0.0]])
