from __future__ import absolute_import

import numpy as np
from scipy.stats import norm

from os import path
import os

from stratified_bayesian_optimization.lib.constant import (
    UNIFORM_FINITE,
    LBFGS_NAME,
    DEBUGGING_DIR,
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
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.util import (
    wrapper_optimization,
    wrapper_objective_voi,
    wrapper_gradient_voi,
)
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class SBO(object):

    _filename = 'opt_sbo_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}.json'.format

    _filename_voi_evaluations = '{iteration}_sbo_{model_type}_{problem_name}_' \
                                '{type_kernel}_{training_name}_{n_training}_{random_seed}.' \
                                'json'.format


    def __init__(self, bayesian_quadrature, discretization_domain=None):
        """

        :param bayesian_quadrature: a bayesian quadrature instance.
        :param discretization_domain: np.array(mxl), discretization of the domain of x.
        """

        self.bq = bayesian_quadrature
        self.discretization = discretization_domain

        self.bounds_opt = self.bq.gp.bounds
        self.opt_separing_domain = False

        # Bounds or list of number of points of the domain
        self.domain_w = [self.bq.gp.bounds[i] for i in self.bq.w_domain]

        if self.bq.distribution == UNIFORM_FINITE:
            self.opt_separing_domain = True

        self.optimization_results = []

    def evaluate(self, point, var_noise=None, mean=None, parameters_kernel=None, cache=True):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True

        :return: float
        """

        if np.any(np.all((self.bq.gp.data['points'] - point[0, :]) == 0, axis=1)):
            return 0.0

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache)

        a = vectors['a']
        b = vectors['b']

        a, b, keep = AffineBreakPointsPrep(a, b)

        keep1, c = AffineBreakPoints(a, b)
        keep1 = keep1.astype(np.int64)

        return self.hvoi(b, c, keep1)


    def evaluate_gradient(self, point, var_noise=None, mean=None, parameters_kernel=None,
                          cache=True):
        """
        Evaluate the acquisition function at the point.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param cache: (boolean) Use cached data and cache data if cache is True

        :return: np.array(n)
        """

        if np.any(np.all((self.bq.gp.data['points'] - point[0, :]) == 0, axis=1)):
            return np.zeros(point.shape[1])

        vectors = self.bq.compute_posterior_parameters_kg(self.discretization, point,
                                                          var_noise=var_noise, mean=mean,
                                                          parameters_kernel=parameters_kernel,
                                                          cache=cache)

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
                                              parameters_kernel=parameters_kernel, cache=cache,
                                              keep_indexes=keep)

        gradient = np.zeros(point.shape[1])
        for i in xrange(point.shape[1]):
            gradient[i] = np.dot(np.diff(gradients[:, i]), evalC)

        return gradient

    def objective_voi(self, point):
        """
        Evaluates the VOI at point.
        :param point: np.array(n)
        :return: float
        """

        point = point.reshape((1, len(point)))
        return self.evaluate(point)

    def grad_obj_voi(self, point):
        """
        Evaluates the gradient of VOI at point.
        :param point: np.array(n)
        :return: np.array(n)
        """

        point = point.reshape((1, len(point)))
        grad = self.evaluate_gradient(point)

        return grad

    def optimize(self, start=None, random_seed=None, parallel=True):
        """
        Optimize the VOI.
        :param start: np.array(n)
        :param random_seed: int
        :param parallel: (boolean) For several tasks, it's run in paralle if it's True

        :return: dictionary with the results of the optimization.
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if start is None:
            start = DomainService.get_points_domain(1, self.bounds_opt,
                                                    type_bounds=self.bq.gp.type_bounds)

        bounds = [tuple(bound) for bound in self.bounds_opt]

        for i in self.bq.w_domain:
            bounds[i] = (None, None)

        optimization = Optimization(
            LBFGS_NAME,
            wrapper_objective_voi,
            bounds,
            wrapper_gradient_voi,
            minimize=False)

        if self.opt_separing_domain:
            start = np.array(start)[:, self.bq.x_domain]
            starts = {}

            max_values = []
            for i, w in enumerate(self.domain_w[0]):
                starts[i] = np.concatenate((start[0, :], np.array([w])))

            args = (False, None, parallel, optimization, self, )
            opt_sol = Parallel.run_function_different_arguments_parallel(
                wrapper_optimization, starts, *args)

            for i in xrange(len(self.domain_w[0])):
                if opt_sol.get(i) is None:
                    logger.info("It wasn't possible to optimize for task %d" % i)
                    continue
                max_values.append(opt_sol[i]['optimal_value'])

            if len(max_values) == 0:
                raise Exception("Optimization failed for all the tasks!")
            index_max = np.argmax(max_values)
            results = opt_sol[index_max]
        else:
            results = optimization.optimize(start, *(self, ))

        self.optimization_results.append(results)

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

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.bq.clean_cache()

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed):
        """
        Write the results of the optimization.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        """
        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        f_name = self._filename(model_type=model_type,
                                problem_name=problem_name,
                                type_kernel=kernel_name,
                                training_name=training_name,
                                n_training=n_training,
                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write(self.optimization_results, debug_path)

    def generate_evaluations(self, problem_name, model_type, training_name, n_training,
                             random_seed, iteration, n_points_by_dimension=None):
        """
        Generates evaluations of the posterior mean, and write them in the debug directory.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param iteration: (int)
        :param n_points_by_dimension: (int) Number of points by dimension

        """

        # TODO: extend to more than one dimension
        bounds = self.bq.gp.bounds
        n_points = n_points_by_dimension
        if n_points is None:
            n_points = (bounds[0][1] - bounds[0][0]) * 10

        points = np.linspace(bounds[0][0], bounds[0][1], n_points)

        values = {}

        if self.bq.tasks:
            for i in xrange(self.bq.n_tasks):
                vals = []
                for point in points:
                    point_ = np.concatenate((np.array([point]), np.array([i])))
                    point_ = point_.reshape((1, len(point_)))
                    value = self.evaluate(point_,)
                    vals.append(value)
                values[i] = vals

        if not os.path.exists(DEBUGGING_DIR):
            os.mkdir(DEBUGGING_DIR)

        debug_dir = path.join(DEBUGGING_DIR, problem_name)

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        kernel_name = ''
        for kernel in self.bq.gp.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]

        f_name = self._filename_voi_evaluations(iteration=iteration,
                                                model_type=model_type,
                                                problem_name=problem_name,
                                                type_kernel=kernel_name,
                                                training_name=training_name,
                                                n_training=n_training,
                                                random_seed=random_seed)

        debug_path = path.join(debug_dir, f_name)

        JSONFile.write({'points': points, 'evaluations': values}, debug_path)
