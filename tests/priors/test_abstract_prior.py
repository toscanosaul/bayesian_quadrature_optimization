import unittest

from nose.tools import raises

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior


class B(AbstractPrior):
    def __init__(self):
        super(B, self).__init__(2)

    def logprob(self, x):
        super(B, self).logprob(x)

    def sample(self, n):
        super(B, self).sample(n)


class TestAbstractPrior(unittest.TestCase):

    @raises(NotImplementedError)
    def test_logprob(self):
        test = B()
        test.logprob(2)

    @raises(NotImplementedError)
    def test_sample(self):
        test = B()
        test.sample(2)
