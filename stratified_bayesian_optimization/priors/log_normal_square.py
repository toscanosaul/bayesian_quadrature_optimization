from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np
from scipy.stats import lognorm


class LogNormalSquare(AbstractPrior):

    def __init__(self, dimension, scale, mean):
        """
        This is the density of Z = Y^2, where Y~log_normal
        :param scale: float
        :param: mean: float
        """

        super(LogNormalSquare, self).__init__(dimension)

        self.scale = scale
        self.mean = mean

    def logprob(self, x):
        """

        :param x: np.array
        :return: float
        """
        if np.any(x < 0):
            return -np.inf

        x = np.sqrt(x)
        dy_dx = 2*x

        return np.sum(lognorm.logpdf(x, self.scale, loc=self.mean)) - np.log(dy_dx)

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        points = np.random.lognormal(mean=self.mean, sigma=self.scale, size=samples)

        return points**2
