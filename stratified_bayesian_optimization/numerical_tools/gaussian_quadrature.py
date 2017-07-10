from __future__ import absolute_import

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class GaussianQuadrature(object):

    def __init__(self, gp_model, tasks, x_domain, distribution):
        self.gp = gp_model
        self.tasks = tasks
        self.x_domain = x_domain
        self.distribution = distribution


    def compute_posterior_parameters(self, points):

        chol, cov = self._chol_cov_including_noise(
            self.var_noise.value[0], self.kernel.hypers_values_as_array)

        y_unbiased = self.data['evaluations'] - self.mean.value[0]
        solve = cho_solve(chol, y_unbiased)

        vec_cov = self.kernel.cross_cov(points, self.data['points'])

        mu_n = self.mean.value[0] + np.dot(vec_cov, solve)

        solve_2 = cho_solve(chol, vec_cov.transpose())
        cov_n = self.kernel.cov(points) - np.dot(vec_cov, solve_2)

        return {
            'mean': mu_n,
            'cov': cov_n,
        }



