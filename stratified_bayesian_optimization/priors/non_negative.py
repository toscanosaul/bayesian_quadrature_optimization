from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np


class NonNegativePrior(AbstractPrior):

    def __init__(self, dimension, prior):
        """
        :param dimension: int
        :param prior: An object from abstract prior
        """
        super(NonNegativePrior, self).__init__(dimension)
        self.prior = prior

    def logprob(self, x):
        """

        :param x: np.array
        :return: float
        """
        if np.any(x <= 0):
            return -np.inf
        else:
            return self.prior.logprob(x)

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        return np.abs(self.prior.sample(samples))
