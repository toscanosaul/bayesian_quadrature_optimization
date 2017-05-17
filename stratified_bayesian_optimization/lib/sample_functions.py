from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.lib.constant import DEFAULT_RANDOM_SEED


class SampleFunctions(object):

    @classmethod
    def sample_from_gp(cls, x, kernel, random_seed=DEFAULT_RANDOM_SEED, n_samples=1):
        """
        Sample function f from GP defined by the kernel.

        :param x: np.array(nxm)
        :param kernel: instance of AbstractKernel
        :param random_seed: int
        :param n_samples: int

        :return: np.array(n_samples x n)
        """

        np.random.seed(random_seed)

        cov = kernel.cov(x)
        mean = np.zeros(x.shape[0])
        f = np.random.multivariate_normal(mean, cov, size=n_samples)

        return f