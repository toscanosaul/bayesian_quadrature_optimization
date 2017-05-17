from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np


# TODO: Check this prior
class HorseShoePrior(AbstractPrior):

    def __init__(self, dimension, scale):
        """

        :param scale: float
        """

        super(HorseShoePrior, self).__init__(dimension)

        self.scale = scale

    def logprob(self, x):
        """

        :param x: np.array
        :return: float
        """
        if np.any(x == 0.0):
            return np.inf

        return np.sum(np.log(np.log(1 + 3.0 * (self.scale / x) ** 2)))

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        lambda_ = np.abs(np.random.standard_cauchy(size=samples))

        return np.random.randn() * lambda_ * self.scale
