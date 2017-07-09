from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
)


class ParameterEntity(object):

    def __init__(self, name, value, prior, bounds=None):
        """

        :param name: str
        :param value: np.array
        :param prior: AbstractPrior
        :param bounds: [((float) min, (float) max)], list with the bounds of each element of the
            array value. If one of them is None, then it means that it's unbounded in that
            direction.

        """
        self.name = name
        self.value = value
        self.prior = prior
        self.dimension = len(self.value)

        if bounds is None:
            bounds = self.get_bounds(self.dimension)

        self.bounds = bounds

    def set_value(self, value):
        self.value = value

    def log_prior(self, value=None):
        if value is None:
            value = self.value

        return self.prior.logprob(value)

    def sample_from_prior(self, n_samples, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        return self.prior.sample(n_samples)

    @staticmethod
    def get_bounds(dimension):
        """
        Get default bounds.

        :param dimension: (int) Dimension of the domain space.

        :return: [[float, float]]
        """

        return dimension * [(SMALLEST_NUMBER, LARGEST_NUMBER)]
