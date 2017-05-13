from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np


class UniformPrior(AbstractPrior):

    def __init__(self, dimension, min, max):
        """

        :param min: [float]
        :param max: [float]
        """

        super(UniformPrior, self).__init__(dimension)

        self.min = np.array(min)
        self.max = np.array(max)

    def logprob(self, x):
        """

        :param x: np.array
        :return: float
        """
        if np.any(x < self.min) or np.any(x > self.max):
            return -np.inf
        else:
            return 0.0

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        return self.min + np.random.rand(samples, self.dimension) * (self.max - self.min)
