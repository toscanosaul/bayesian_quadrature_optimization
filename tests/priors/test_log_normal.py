from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.priors.log_normal import LogNormal


class TestUniformPrior(unittest.TestCase):

    def setUp(self):
        self.prior = LogNormal(1, [1], [0])

    def test_logprob(self):
        assert self.prior.logprob(np.array([[-1]])) == -np.inf

        result = 1.0 / (np.sqrt(2.0 * np.pi))
        assert self.prior.logprob(np.array([[1]])) == np.log(result)

        result = 1.0 / (np.sqrt(2.0 * np.pi))
        assert self.prior.logprob(np.array([1])) == np.log(result)

    def test_sample(self):
        np.random.seed(1)
        sample = self.prior.sample(1)
        assert self.prior.sample(1, 1) == sample
