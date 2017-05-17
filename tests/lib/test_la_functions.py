import unittest

import numpy.testing as npt
import scipy.linalg as spla
from scipy import linalg

from stratified_bayesian_optimization.lib.la_functions import *
from stratified_bayesian_optimization.kernels.matern52 import Matern52


class TestLAFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)
        n_points = 50
        points = np.linspace(0, 500, n_points)
        points = points.reshape([n_points, 1])
        kernel = Matern52.define_kernel_from_array(1, np.array([100.0, 1.0]))
        self.cov = kernel.cov(points)

        self.cov_2 = np.array(
            [[3.76518160e-02, 8.35508788e-03, 2.26375310e-03, 4.81839112e-02, 1.19018900e-02],
             [8.35508788e-03, 3.76518160e-02, 4.65867508e-05, 2.23451418e-03, 3.20904593e-04],
             [2.26375310e-03, 4.65867508e-05, 3.76518160e-02, 8.45452922e-03, 3.79688244e-02],
             [4.81839112e-02, 2.23451418e-03, 8.45452922e-03, 3.76518160e-02, 3.53194132e-02],
             [1.19018900e-02, 3.20904593e-04, 3.79688244e-02, 3.53194132e-02, 3.76518160e-02]])

        self.cov_ = np.array(
            [[1.04681851e+00, 9.95986475e-02, 2.71028578e-02, 5.70675240e-01, 1.41705272e-01],
             [9.95986475e-02, 1.04681851e+00, 5.64490356e-04, 2.67539188e-02, 3.86574119e-03],
             [2.71028578e-02, 5.64490356e-04, 1.04681851e+00, 1.00779972e-01, 4.50126283e-01],
             [5.70675240e-01, 2.67539188e-02, 1.00779972e-01, 1.04681851e+00, 4.18836616e-01],
             [1.41705272e-01, 3.86574119e-03, 4.50126283e-01, 4.18836616e-01, 1.04681851e+00]])

    def test_cholesky(self):
        chol = cholesky(self.cov)
        npt.assert_almost_equal(np.dot(chol, chol.transpose()), self.cov)

        with self.assertRaises(linalg.LinAlgError):
            cholesky(self.cov_2)

        chol_ = cholesky(self.cov_2, max_tries=7)
        npt.assert_almost_equal(self.cov_2, np.dot(chol_, chol_.transpose()), decimal=1)

        with self.assertRaises(linalg.LinAlgError):
            cholesky(np.array([[-1, 5], [3, 7]]))

    def test_cho_solve(self):
        chol = cholesky(self.cov)
        y = np.linspace(1.0, 100.0, self.cov.shape[0])
        sol = cho_solve(chol, y)
        npt.assert_almost_equal(np.dot(self.cov, sol), y)
