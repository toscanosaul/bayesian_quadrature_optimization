from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np
from scipy.stats import lognorm


class LogNormal(AbstractPrior):

    def __init__(self, dimension, scale, mu):
        """
        See https://en.wikipedia.org/wiki/Log-normal_distribution

        :param scale: [float]
        :param: mean: [float]
        """

        super(LogNormal, self).__init__(dimension)

        self.scale = scale
        self.mu = mu

    def logprob(self, x):
        """

        :param x: np.array(nxm)
        :return: float
        """

        if len(x.shape) == 1:
            x = x.reshape(len(x), 1)

        if np.any(x < 0):
            return -np.inf

        llh = 0
        for dim in xrange(self.dimension):
            llh += np.sum(lognorm.logpdf(x[:, dim], s=self.scale[dim], scale=np.exp(self.mu[dim])))
        return llh

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        points = np.zerps(samples, self.dimension)

        for dim in xrange(self.dimension):
            points[:, dim] = lognorm.rvs(s=self.scale[dim], scale=np.exp(self.mu[dim]),
                                         size=samples)

        return points
