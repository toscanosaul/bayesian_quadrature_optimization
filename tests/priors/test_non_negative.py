from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.priors.non_negative import NonNegativePrior
from stratified_bayesian_optimization.priors.log_normal_square import LogNormalSquare


class TestUniformPrior(unittest.TestCase):

    def setUp(self):
        self.dimension = 1
        self.scale = 1.0
        self.mu = 1.0
        self.prior_ = LogNormalSquare(self.dimension, self.scale, self.mu)
        self.prior = NonNegativePrior(1, self.prior_)

    def test_logprob(self):
        assert self.prior.logprob(0) == -np.inf

    def test_sample(self):
        np.random.seed(1)
        sample = self.prior.sample(1)
        assert self.prior.sample(1, 1) == sample