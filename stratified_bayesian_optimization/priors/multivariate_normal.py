from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np
from scipy.stats import multivariate_normal


class MultivariateNormalPrior(AbstractPrior):

    def __init__(self, dimension, mu, cov):
        """

        :param dimension: int
        :param mu: np.array(n)
        :param cov: np.array(nxn)
        """
        super(MultivariateNormalPrior, self).__init__(dimension)
        self.mu = mu
        self.cov = cov

    def logprob(self, x):
        """
        :param x: np.array(kxn)
        :return: float
        """

        return np.sum(multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov))

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        return np.random.multivariate_normal(self.mu, self.cov, size=samples).T.squeeze()
