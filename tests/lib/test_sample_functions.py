import unittest

import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.lib.sample_functions import SampleFunctions
from stratified_bayesian_optimization.kernels.matern52 import Matern52


class TestSampleFunctions(unittest.TestCase):

    def test_sample_from_gp(self):
        x = np.linspace(0, 10, 50)
        x = x.reshape([50,1])
        kernel = Matern52.define_kernel_from_array(1, np.array([3.0, 5.0]))
        function = SampleFunctions.sample_from_gp(x, kernel, n_samples=100000)

        mean = np.mean(function, axis=0)
        cov = np.cov(function.transpose())
        cov_ = kernel.cov(x)

        npt.assert_almost_equal(mean, np.zeros(len(mean)), decimal=1)
        npt.assert_almost_equal(cov, cov_, decimal=1)
