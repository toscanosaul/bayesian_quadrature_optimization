from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.priors.multivariate_normal import MultivariateNormalPrior


class TestUniformPrior(unittest.TestCase):

    def setUp(self):
        self.dimension = 1
        self.scale = np.array([[1.0]])
        self.mu = np.array([0.0])
        self.prior = MultivariateNormalPrior(self.dimension, self.mu, self.scale)

    def test_logprob(self):
        result = 1.0 / (np.sqrt(2.0 * np.pi))
        assert self.prior.logprob([0]) == np.log(result)

    def test_sample(self):
        np.random.seed(1)
        sample = self.prior.sample(1)
        assert self.prior.sample(1, 1) == sample
