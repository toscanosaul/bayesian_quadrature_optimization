from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from stratified_bayesian_optimization.lib.constant import (
    UNIFORM_FINITE,
    LBFGS_NAME,
)
from stratified_bayesian_optimization.lib.affine_break_points import (
    AffineBreakPointsPrep,
    AffineBreakPoints,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.parallel import Parallel


class SBO(object):

    def __init__(self, bayesian_quadrature, discretization_domain=None):
        """

        :param bayesian_quadrature: a bayesian quadrature instance.
        :param discretization_domain: np.array(mxl), discretization of the domain of x.
        """

        self.bq = bayesian_quadrature
        self.discretization = discretization_domain

        self.bounds_opt = self.bq.gp.bounds
        self.opt_separing_domain = False

        self.domain_w = [self.bq.gp.bounds[i] for i in self.bq.w_domain]

        if self.bq.distribution == UNIFORM_FINITE:
            self.opt_separing_domain = True

        if self.bq.distribution == UNIFORM_FINITE:
            bounds_x = [self.bounds_opt[i] for i in xrange(len(self.bounds_opt)) if i in
                        self.bq.x_domain]
            self.bounds_opt = bounds_x



    def evaluate(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)

        :return: float
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel)

        a = vectors['a']
        b = vectors['b']

        a, b, keep = AffineBreakPointsPrep(a, b)
        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)

        return self.hvoi(b, c, keep1)


    def evaluate_gradient(self, point, var_noise=None, mean=None, parameters_kernel=None):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)

        :return: np.array(n)
        """

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel)

        a = vectors['a']
        b = vectors['b']

        a, b, keep = AffineBreakPointsPrep(a, b)
        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)
        M = len(keep1)

        if M <=1 :
            return np.zeros(point.shape[1])

        keep=keep[keep1] #indices conserved

        c=c[keep1+1]
        c2=np.abs(c[0:M-1])
        evalC=norm.pdf(c2)

        gradients = self.bq.gradient_vector_b(point, self.discretization[keep, :],
                                              var_noise=var_noise, mean=mean,
                                              parameters_kernel=parameters_kernel)

        gradient = np.zeros(point.shape[1])
        for i in xrange(point.shape[1]):
            gradient[i] = np.dot(np.diff(gradients[:, i]), evalC)

        return gradient

    def objective_voi(self, point, *args):
        """
        Evaluates the VOI at point.
        :param point: np.array(n)
        :param args: additional part of point if needed. For example,
            x = point, and args[0] is w, so the function is evaluated at (x, w)
        :return: float
        """
        if self.opt_separing_domain:
            point = np.concatenate((point, args[0]))
        point = point.reshape((1, len(point)))
        return self.evaluate(point)

    def grad_obj_voi(self, point, *args):
        """
        Evaluates the gradient of VOI at point.
        :param point: np.array(n)
        :param args: additional part of point if needed
        :return: np.array(n)
        """
        if self.opt_separing_domain:
            point = np.concatenate((point, args[0]))
        point = point.reshape((1, len(point)))
        return self.evaluate_gradient(point)

    def optimize(self, start=None, random_seed=None):
        """
        Optimize the VOI.
        :param start: np.array(n)
        :param random_seed: int

        :return: dictionary with the results of the optimization.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if start is None:
            start = DomainService.get_points_domain(1, self.bounds_opt,
                                                    type_bounds=len(self.bounds_opt) * [0])

        bounds = [tuple(bound) for bound in self.bounds_opt]

        objective_function = self.objective_voi
        grad_function = self.grad_obj_voi

        if self.opt_separing_domain:
            objective_functions = {}
            grad_functions = {}
            optimizations = {}
            parallel_functions = {}
            for i, w_point in enumerate(self.domain_w):
                f = lambda x: self.objective_voi(x, *(np.array(w_point),))
                objective_functions[i] = f

                grad_f = lambda x: self.grad_obj_voi(x, *(np.array(w_point),))
                grad_functions[i] = grad_f

                optimization = Optimization(
                    LBFGS_NAME,
                    objective_functions[i],
                    bounds,
                    objective_functions[i],
                    minimize=False)
                optimizations[i] = optimization

       #     new_gp_objects = Parallel.run_function_different_arguments_parallel(
        #        wrapper_fit_gp_regression, gp_objects, all_success=False, **kwargs)
        else:
            optimization = Optimization(
                LBFGS_NAME,
                objective_function,
                bounds,
                grad_function,
                minimize=False)

            results = optimization.optimize(start)
        return results


    @staticmethod
    def hvoi (b,c,keep):
        M=len(keep)
        if M>1:
            c=c[keep+1]
            c2=-np.abs(c[0:M-1])
            tmp=norm.pdf(c2)+c2*norm.cdf(c2)
            return np.sum(np.diff(b[keep])*tmp)
        else:
            return 0