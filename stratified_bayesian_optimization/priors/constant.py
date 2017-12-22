from __future__ import absolute_import

from stratified_bayesian_optimization.priors.abstract_prior import AbstractPrior

import numpy as np
from scipy.stats import norm


class Constant(AbstractPrior):

    def __init__(self, dimension, value):
        """

        :param dimension: int
        :param value: float
        """
        super(Constant, self).__init__(dimension)
        self.constant = value

    def logprob(self, x):
        """
        :param x: np.array(n)
        :return: float
        """
        if self.constant == x[0]:
            return 0.0
        else:
            return - np.inf

    def sample(self, samples, random_seed=None):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array(samples, self.dimension)
        """
        z = samples * [self.constant]
        z = np.array(z)
        z = z.reshape((len(z), 1))
        return z
