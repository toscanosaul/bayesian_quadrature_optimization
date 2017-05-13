from __future__ import absolute_import

import unittest

import mock

import numpy as np

from stratified_bayesian_optimization.priors.uniform import UniformPrior


class TestUniformPrior(unittest.TestCase):

    def setUp(self):
        self.dimension = 1
        self.min = np.array([0.0])
        self.max = np.array([1.0])
        self.uniform_prior = UniformPrior(self.dimension, self.min, self.max)

    def test_logprob(self):
        assert self.uniform_prior.logprob(-1) == -np.inf
        assert self.uniform_prior.logprob(0.5) == 0.0

    @mock.patch('numpy.random.rand')
    def test_sample(self, random_call):
        random_call.return_value = 0.5

        assert self.uniform_prior.sample(1) == 0.5
        assert self.uniform_prior.sample(1, 2) == 0.5

