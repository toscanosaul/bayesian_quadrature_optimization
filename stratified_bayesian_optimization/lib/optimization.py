from __future__ import absolute_import

from scipy.optimize import fmin_l_bfgs_b

import numpy as np

from stratified_bayesian_optimization.lib.optimization_methods import (
    newton_cg, trust_ncg, dogleg, nelder_mead)
from stratified_bayesian_optimization.lib.constant import (
    LBFGS_NAME, SGD_NAME, NEWTON_CG_NAME, TRUST_N_CG, DOGLEG, NELDER)
from stratified_bayesian_optimization.lib.stochastic_gradient_descent import SGD

from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class Optimization(object):

    _gradient_free_ = [NELDER]
    _optimizers_ = [LBFGS_NAME, SGD_NAME, NEWTON_CG_NAME, TRUST_N_CG, DOGLEG]
    _hessian_methods = [NEWTON_CG_NAME, TRUST_N_CG, DOGLEG]

    def __init__(self, optimizer_name, function, bounds, grad, hessian=None, minimize=True,
                 full_gradient=None, debug=True, args=None, tol=None, **kwargs):
        """
        Class used to minimize function.

        :param optimizer_name: str
        :param function:
        :param bounds: [(min, max)] for each point
        :param grad:
        :param hessian: function that computes the hessian
        :param minimize: boolean
        :param full_gradient: function that computes the complete gradient. Used in SGD.
        :param debug: boolean
        :param args: () additional arguments for the full_gradient function
        :parma tol: float
        :param kwargs:
            -'factr': int
            -'maxiter': int
        """
        self.optimizer_name = optimizer_name
        self.optimizer = self._get_optimizer(optimizer_name)
        self.optimizer_default = self._get_optimizer(LBFGS_NAME)
        self.function = function
        self.gradient = grad
        self.bounds = bounds
        self.dim = len(self.bounds)
        self.minimize = minimize
        self.optimization_options = kwargs
        self.args = args
        self.debug = debug
        self.full_gradient = full_gradient
        self.hessian = hessian
        self.tol = tol


    @staticmethod
    def _get_optimizer(optimizer_name):
        """

        :param optimizer_name: (str)
        :return: optimizer function
        """

        if optimizer_name == LBFGS_NAME:
            return fmin_l_bfgs_b

        if optimizer_name == SGD_NAME:
            return SGD

        if optimizer_name == NEWTON_CG_NAME:
            return newton_cg

        if optimizer_name == TRUST_N_CG:
            return trust_ncg

        if optimizer_name == DOGLEG:
            return dogleg

        if optimizer_name == NELDER:
            return nelder_mead

    def optimize(self, start, *args):
        """

        :param start: (np.array(n)) starting point of the optimization of the llh.
        :param args: Arguments to pass to function and gradient.

        :return: {
            'solution': np.array(n),
            'optimal_value': float,
            'gradient': np.array(n),
            'warnflag': int,
            'task': str
        }
        """

        if self.minimize:
            if self.optimizer_name == NEWTON_CG_NAME:
                opt = self.optimizer(self.function, start, fprime=self.gradient,
                                     hessian=self.hessian, args=args,
                                     bounds=self.bounds, **self.optimization_options)
            else:
                opt = self.optimizer(self.function, start, fprime=self.gradient, args=args,
                                     bounds=self.bounds, **self.optimization_options)
        else:
            def f(x, *args):
                return -1.0 * self.function(x, *args)

            if self.gradient is not None:
                def grad(x, *args):
                    return -1.0 * self.gradient(x, *args)

            if self.hessian is not None:
                def hessian(x, *args):
                    return -1.0 * self.hessian(x, *args)

            if self.optimizer_name in self._hessian_methods:
                try:
                    opt = self.optimizer(
                        f, start,
                        fprime=grad, hessian=hessian,
                        args=args, tol=self.tol,
                        bounds=self.bounds, **self.optimization_options)
                except Exception as e:
                    opt = self.optimizer_default(
                        f, start,
                        fprime=grad,
                        args=args,
                        bounds=self.bounds, **self.optimization_options)
            elif self.optimizer_name in self._gradient_free_:
                opt = self.optimizer(
                    f, start,
                    args=args,
                    bounds=self.bounds)
            else:
                opt = self.optimizer(
                    f, start,
                    fprime=grad,
                    args=args,
                    bounds=self.bounds, **self.optimization_options)

        return {
            'solution': opt[0],
            'optimal_value': opt[1] if self.minimize else -1.0 * opt[1],
            'gradient': opt[2]['grad'] if self.minimize else -1.0 * opt[2]['grad'],
            'warnflag': opt[2]['warnflag'],
            'task': opt[2]['task'],
            'nit': opt[2]['nit'],
            'funcalls': opt[2]['funcalls'],
        }


    def SGD(self, start, n, *args, **kwargs):
        if not self.minimize:
            def grad(x, *args, **kwargs):
                return -1.0 * self.gradient(x, *args, **kwargs)
        else:
            grad = self.gradient

        opt = self.optimizer(
            start,
            grad,
            n,
            args=args,
            kwargs=kwargs,
            bounds=self.bounds,
            **self.optimization_options
        )

        value_objective = None
        gradient = None

        if self.debug:
            value_objective = self.function(opt, *self.args)
            gradient = self.full_gradient(opt, *self.args)
            if gradient is np.nan:
                gradient = 'unavailable'

        return {
            'solution': opt,
            'optimal_value': value_objective,
            'gradient': gradient,
        }
