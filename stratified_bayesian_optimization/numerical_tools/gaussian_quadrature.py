from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import (
    SOL_CHOL_Y_UNBIASED,
    UNIFORM_FINITE,
    TASKS,
)
from stratified_bayesian_optimization.lib.la_functions import (
    cho_solve,
)
from stratified_bayesian_optimization.lib.expectations import (
    uniform_finite,
)

logger = SBOLog(__name__)


class GaussianQuadrature(object):

    _expectations_map = {
        UNIFORM_FINITE: {
            'expectation': uniform_finite,
            'parameter': TASKS,
        },
    }

    def __init__(self, gp_model, x_domain, distribution, parameters_distribution):
        """

        :param gp_model: gp_fitting_gaussian instance
        :param x_domain: [int], indices of the x domain
        :param distribution: (str), it must be in the list of distributions:
            [UNIFORM_FINITE]
        :param parameters_distribution: (dict) dictionary with parameters of the distribution.
            -UNIFORM_FINITE: TASKS
        """
        self.gp = gp_model
        self.parameters_distribution = parameters_distribution
        self.dimension_domain = self.gp.dimension_domain
        self.x_domain = x_domain
        self.w_domain = [i for i in range(self.gp.dimension_domain) if i not in x_domain]
        self.expectation = self._expectations_map[distribution]
        self.arguments_expectation = {}

        if self.expectation['parameter'] == TASKS:
            n_tasks = self.parameters_distribution.get(TASKS)
            self.arguments_expectation['domain_random'] = np.arange(n_tasks).reshape((n_tasks, 1))

    def evaluate_quadrature_cross_cov(self, point, points_2, parameters_kernel):
        """
        Evaluate the quadrature cross cov respect to points_1, i.e.
            Expectation(cov((x_i,w_i), (x'_j,w'_j))) respect to w_i, where points_1 = (x_i), and
            points_2 = (x'_j, w'_j).
        This is [B(x, j)] in the SBO paper.

        :param point: np.array(1xk)
        :param points_2: np.array(mxk')
        :param parameters_kernel: np.array(l)
        :return: np.array(m)
        """

        f = lambda x: self.gp.evaluate_cross_cov(x, points_2, parameters_kernel)

        parameters = {
            'f': f,
            'point': point,
            'index_points': self.x_domain,
            'index_random': self.w_domain,
        }

        parameters.update(self.arguments_expectation)

        B = self.expectation['expectation'](**parameters)

        return B

    def compute_posterior_parameters(self, points, candidate_point, var_noise=None, mean=None,
                                     parameters_kernel=None):
        """
        Compute posterior parameters of the GP after integrating out the random parameters.

        :param points: np.array(nxk)
        :param candidate_point: np.array(1xm), (new_x, new_w)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)

        :return: {
            'mean': np.array(n),
            'cov': np.array(nxn)
        }
        """

        if var_noise is None:
            var_noise = self.gp.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.gp.kernel.hypers_values_as_array

        if mean is None:
            mean = self.gp.mean.value[0]

        chol, cov = self.gp._chol_cov_including_noise(
            var_noise, parameters_kernel)

        y_unbiased = self.gp.data['evaluations'] - mean

        cached_solve = self.gp._get_cached_data((var_noise, tuple(parameters_kernel), mean),
                                             SOL_CHOL_Y_UNBIASED)

        if cached_solve is False:
            solve = cho_solve(chol, y_unbiased)
            self.gp._updated_cached_data((var_noise, tuple(parameters_kernel), mean), solve,
                                      SOL_CHOL_Y_UNBIASED)
        else:
            solve = cached_solve

        n = points.shape[0]
        m = self.gp.data['points'].shape[0]
        vec_covs = np.zeros((n, m))
        b_new = np.zeros((n, 1))
        for i in xrange(n):
            vec_covs[i, :] = self.evaluate_quadrature_cross_cov(
                points[i:i+1,:], self.gp.data['points'], parameters_kernel)

            b_new[i, 0] = self.evaluate_quadrature_cross_cov(
                points[i:i+1,:], candidate_point, parameters_kernel)

        mu_n = mean + np.dot(vec_covs, solve)

        # TODO: CACHE SO WE DON'T COMPUTE MU_N ALL THE TIME
        cross_cov = self.gp.evaluate_cross_cov(candidate_point, self.gp.data['points'],
                                                parameters_kernel)

        solve_2 = cho_solve(chol, cross_cov[0, :])
        numerator = b_new[:, 0] - np.dot(vec_covs, solve_2)

        new_cross_cov = self.gp.evaluate_cross_cov(candidate_point, candidate_point,
                                                   parameters_kernel)

        denominator = new_cross_cov - np.dot(cross_cov, solve_2)
        denominator = denominator[0, 0]

        return {
            'mean': mu_n,
            'cov': numerator ** 2 / denominator,
        }
