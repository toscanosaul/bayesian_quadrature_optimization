from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np
from scipy.stats import norm


class GaussianPrior(AbstractPrior):

    def __init__(self, dimension, mu, sigma):
        """

        :param dimension: int
        :param mu: float
        :param sigma: float
        """
        super(GaussianPrior, self).__init__(dimension)
        self.mu = mu
        self.sigma = sigma

    def logprob(self, x):
        """
        :param x: np.array(n)
        :return: float
        """

        return np.sum(norm.logpdf(x, loc=self.mu, scale=self.sigma))

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        return self.mu + np.random.randn(samples, self.dimension) * self.sigma
