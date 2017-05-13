from __future__ import absolute_import

from abc import ABCMeta, abstractmethod


class AbstractPrior(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimension):
        """

        :param dimension: int
        """
        self.dimension = dimension

    @abstractmethod
    def logprob(self, x):
        """

        :param x: np.array
        :return: float
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def sample(self, samples, random_seed):
        """

        :param samples: int
        :param random_seed: int
        :return: np.array
        """
        raise NotImplementedError("Not implemented")
