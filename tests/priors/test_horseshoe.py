from __future__ import absolute_import

import unittest

import mock

import numpy as np

from stratified_bayesian_optimization.priors.horseshoe import HorseShoePrior


class TestHorseShoePrior(unittest.TestCase):

    def setUp(self):
        self.dimension = 1
        self.scale = 1.0

        self.prior = HorseShoePrior(self.dimension, self.scale)

    def test_logprob(self):
        assert self.prior.logprob(0.0) == np.inf

    def test_sample(self):
        np.random.seed(1)
        sample = self.prior.sample(1)
        assert self.prior.sample(1, 1) == sample