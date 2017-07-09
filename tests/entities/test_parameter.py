from __future__ import absolute_import

import numpy as np

import unittest

from mock import create_autospec
from doubles import expect

from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
)


class TestParameterEntity(unittest.TestCase):

    def setUp(self):
        self.name = 'test'
        self.value = np.array([1, 2])
        self.prior = create_autospec(UniformPrior)
        self.parameter = ParameterEntity(self.name, self.value, self.prior)

    def test_set_value(self):
        self.parameter.set_value(np.array([3, 4]))
        assert np.all(self.parameter.value) == np.all(np.array([3, 4]))

    def test_log_prior(self):
        expect(self.prior).logprob.once().and_return(0.0)
        assert 0.0 == self.parameter.log_prior()

    def test_sample_from_prior(self):
        expect(self.prior).sample.twice().and_return(0.0)
        assert 0.0 == self.parameter.sample_from_prior(1)
        assert 0.0 == self.parameter.sample_from_prior(1, 1)

    def test_get_bounds(self):
        assert self.parameter.get_bounds(2) == 2 * [(SMALLEST_NUMBER, LARGEST_NUMBER)]
