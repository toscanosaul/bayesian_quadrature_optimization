from __future__ import absolute_import

from copy import deepcopy

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
        self.bounds = self.process_bounds(bounds)

    def set_value(self, value):
        self.value = value

    def log_prior(self, value=None):
        if value is None:
            value = self.value

        return self.prior.logprob(value)

    def sample_from_prior(self, n_samples):
        return self.prior.sample(n_samples)

    def process_bounds(self, bounds):
        """
        Replace None in bounds by floats. If a lower bound is None, then it's replaced by
        _lower_bound. If a upper bound is None, then it's replaced by _upper_bound.

        :param bounds: [(float/None, float/None)]
        :return: [(float, float)]
        """

        if bounds is None:
            return self.dimension *  [(SMALLEST_NUMBER, LARGEST_NUMBER)]

        new_bounds = deepcopy(bounds)

        for index, bound in enumerate(bounds):
            if bound[0] is None:
                new_bounds[index][0] = SMALLEST_NUMBER
            if bounds[1] is None:
                new_bounds[index][1] = LARGEST_NUMBER

        return new_bounds
