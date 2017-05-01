from __future__ import absolute_import

from scipy.optimize import fmin_l_bfgs_b

from stratified_bayesian_optimization.lib.constant import LBFGS_NAME


class Optimization(object):

    _optimizers_ = [LBFGS_NAME]

    def __init__(self, optimizer_name, function, bounds, grad, minimize=True):
        """
        Class used to minimize function.

        :param optimizer_name: str
        :param function:
        :param bounds: [(min, max)] for each point
        :param grad:
        :param minimize: boolean
        """
        self.optimizer_name = optimizer_name
        self.optimizer = self._get_optimizer(optimizer_name)
        self.function = function
        self.gradient = grad
        self.bounds = bounds
        self.minimize = minimize

    @staticmethod
    def _get_optimizer(optimizer_name):
        """

        :param optimizer_name: (str)
        :return: optimizer function
        """

        if optimizer_name == LBFGS_NAME:
            return fmin_l_bfgs_b

    def optimize(self, start):
        """

        :param start: (np.array(n)) starting point of the optimization of the llh.

        :return: {
            'solution': np.array(n),
            'optimal_value': float,
            'gradient': np.array(n),
            'warnflag': int,
            'task': str
        }
        """
        if self.minimize:
            opt = self.optimizer(self.function, start, fprime=self.gradient, bounds=self.bounds)
        else:
            opt = self.optimizer(
                lambda x: -1.0 * self.function(x), start,
                fprime=lambda x: -1.0 * self.gradient(x),
                bounds=self.bounds)

        return {
            'solution': opt[0],
            'optimal_value': opt[1] if self.minimize else -1.0 * opt[1],
            'gradient': opt[2]['grad'] if self.minimize else -1.0 * opt[1],
            'warnflag': opt[2]['warnflag'],
            'task': opt[2]['task']
        }
