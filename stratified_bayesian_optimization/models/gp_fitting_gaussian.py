from __future__ import absolute_import

import scipy.linalg as spla
import numpy as np

from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    MEAN_NAME,
    VAR_NOISE_NAME,
    CHOL_COV,
    SOL_CHOL_Y_UNBIASED,
)
from stratified_bayesian_optimization.lib.util_gp_fitting import (
    get_kernel_default,
    get_kernel_class,
)
from stratified_bayesian_optimization.lib.util import (
    separate_numpy_arrays_in_lists,
    wrapper_fit_gp_regression,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.constant import LBFGS_NAME
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling

logger = SBOLog(__name__)


class GPFittingGaussian(object):

    _possible_kernels_ = [MATERN52_NAME, TASKS_KERNEL_NAME, PRODUCT_KERNELS_SEPARABLE]

    def __init__(self, type_kernel, training_data, dimensions, thinning=0):
        """
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param thinning: (int)
        """

        self.type_kernel = type_kernel
        self.class_kernel = get_kernel_class(type_kernel)
        self.training_data = training_data
        self.dimensions = dimensions

        self.kernel = get_kernel_default(type_kernel, self.dimensions)

        self.mean = ParameterEntity(
            MEAN_NAME, np.array([0]), UniformPrior(1, [-1e10], [1e10]))
        self.var_noise = ParameterEntity(
            VAR_NOISE_NAME, np.array([1e-10]), UniformPrior(1, [1e-10], [1e10]))

        self.thinning = thinning
        self.slice_sampler = SliceSampling(self.log_prob_parameters)

        self.cache_chol_cov = {}
        self.cache_sol_chol_y_unbiased = {}

    @property
    def get_parameters_model(self):
        """

        :return: ([ParameterEntity]) list with the parameters of the model
        """
        return [self.var_noise, self.mean] + self.kernel.hypers_as_list

    @property
    def get_value_parameters_model(self):
        """

        :return: np.array(n)
        """
        parameters = self.get_parameters_model
        values = []
        for parameter in parameters:
            values.append(parameter.value)

        return np.concatenate(values)

    def _get_cached_data(self, index, name):
        """

        :param index: tuple associated to the type.
            -(var_noise, parameters_kernel) if name is CHOL_COV
            -(var_noise, parameters_kernel, mean) if name is SOL_CHOL_Y_UNBIASED
        :param name: (str) SOL_CHOL_Y_UNBIASED or CHOL_COV
        :return: cached data if it's cached, otherwise False
            -(chol, cov) if name is CHOL_COV
            -cov^-1 (y-mean) if name is SOL_CHOL_Y_UNBIASED
        """
        if name == CHOL_COV:
            if index in self.cache_chol_cov:
                return self.cache_chol_cov[index]
        if name == SOL_CHOL_Y_UNBIASED:
            if index in self.cache_sol_chol_y_unbiased:
                return self.cache_sol_chol_y_unbiased[index]
        return False

    def _updated_cached_data(self, index, value, name):
        """

        :param index: tuple associated to the type.
            -(var_noise, parameters_kernel) if CHOL_COV
            -(var_noise, parameters_kernel, mean) if SOL_CHOL_Y_UNBIASED
        :param value: value to be cached
        :param name: (str) SOL_CHOL_Y_UNBIASED or CHOL_COV

        """
        if name == CHOL_COV:
            self.cache_chol_cov = {}
            self.cache_sol_chol_y_unbiased = {}
            self.cache_chol_cov[index] = value
        if name == SOL_CHOL_Y_UNBIASED:
            self.cache_sol_chol_y_unbiased = {}
            self.cache_sol_chol_y_unbiased[index] = value

    def _chol_cov_including_noise(self, var_noise, parameters_kernel):
        """
        Compute the Cholesky decomposition of
        covariance = cov_kernel + np.diag(var_noise_observations) + np.diag(var_noise), and the
        covariance matrix
        :param var_noise: float
        :param parameters_kernel: np.array(k)
        :return: np.array(nxn) (chol), np.array(nxn) (cov)
        """

        cached = self._get_cached_data((var_noise, parameters_kernel), CHOL_COV)
        if cached is not False:
            return cached

        n = self.training_data['points'].shape[0]

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.dimensions[0]),
                separate_numpy_arrays_in_lists(self.training_data['points'], self.dimensions[0]),
                self.dimensions, self.type_kernel[1: ])
        else:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                parameters_kernel, self.training_data['points'], self.dimensions[0]
            )

        if self.training_data.get('var_noise'):
            cov += np.diag(self.training_data['var_noise'])

        cov += np.diag(var_noise * np.ones(n))

        chol = spla.cholesky(cov, lower=True)

        self._updated_cached_data((var_noise, parameters_kernel), (chol, cov), CHOL_COV)

        return chol, cov

    def log_likelihood(self, var_noise, mean, parameters_kernel):
        """
        GP log likelihood: y(x) ~ f(x) + epsilon, where epsilon(x) are iid N(0,var_noise), and
        f(x) ~ GP(mean, cov)

        :param var_noise: (float) variance of the noise
        :param mean: (float)
        :param parameters_kernel: np.array(k), The order of the parameters is given in the
            definition of the class kernel.
        :return: float

        """
        chol, cov = self._chol_cov_including_noise(var_noise, parameters_kernel)
        y_unbiased = self.training_data['evaluations'] - mean

        cached_solve = self._get_cached_data((var_noise, parameters_kernel, mean),
                                             SOL_CHOL_Y_UNBIASED)

        if cached_solve is False:
            solve = spla.cho_solve((chol, True), y_unbiased)
            self._updated_cached_data((var_noise, parameters_kernel, mean), solve,
                                      SOL_CHOL_Y_UNBIASED)
        else:
            solve = cached_solve

        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(y_unbiased, solve)

    def grad_log_likelihood_dict(self, var_noise, mean, parameters_kernel):
        """
        Computes the gradient of the log likelihood

        :param var_noise: (float) variance of the noise
        :param mean: (float)
        :param parameters_kernel: np.array(k), The order of the parameters is given in the
            definition of the class kernel.
        :return: {'var_noise': float, 'mean': float, 'kernel_params': np.array(n)}
        """

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.dimensions[0]),
                separate_numpy_arrays_in_lists(self.training_data['points'], self.dimensions[0]),
                self.dimensions, self.type_kernel[1: ])
        else:
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                parameters_kernel, self.training_data['points'], self.dimensions[0])

        chol, cov = self._chol_cov_including_noise(var_noise, parameters_kernel)

        y_unbiased = self.training_data['evaluations'] - mean

        cached_solve = self._get_cached_data((var_noise, parameters_kernel, mean),
                                             SOL_CHOL_Y_UNBIASED)
        if cached_solve is False:
            solve = spla.cho_solve((chol, True), y_unbiased)
            self._updated_cached_data((var_noise, parameters_kernel, mean), solve,
                                      SOL_CHOL_Y_UNBIASED)
        else:
            solve = cached_solve

        gradient_kernel_params = np.zeros(len(parameters_kernel))
        for i in xrange(len(parameters_kernel)):
            gradient_kernel_params[i] = GradientGPFittingGaussian.\
                compute_gradient_llh_given_grad_cov(grad_cov[i], chol, solve)

        n_training_points = self.training_data.shape[0]

        gradient = {}
        gradient['kernel_params'] = gradient_kernel_params
        gradient['mean'] = GradientGPFittingGaussian.compute_gradient_mean(
            chol, y_unbiased, n_training_points)

        grad_var_noise = GradientGPFittingGaussian.compute_gradient_kernel_respect_to_noise(
            n_training_points)
        gradient['var_noise'] = GradientGPFittingGaussian.compute_gradient_llh_given_grad_cov(
            grad_var_noise, cov, solve)

        return gradient

    def grad_log_likelihood(self, var_noise, mean, parameters_kernel):
        """
        Computes the gradient of the log likelihood

        :param var_noise: (float) variance of the noise
        :param mean: (float)
        :param parameters_kernel: np.array(k), The order of the parameters is given in the
            definition of the class kernel.
        :return: (np.array(number_parameters), the first part is the derivative respect to
            var_noise, the second part respect to the mean, and the last part respect to the
            parameters of the kernel.
        """

        gradient = self.grad_log_likelihood_dict(var_noise, mean, parameters_kernel)

        gradient_array = np.zeros(2 + len(parameters_kernel))
        gradient_array[0] = gradient['var_noise']
        gradient_array[1] = gradient['mean']
        gradient_array[2:] = gradient['kernel_params']

        return gradient_array

    def sample_parameters_prior(self, n_samples):
        """
        Sample parameters of the GP model from their prior

        :param n_samples: int
        :return: np.array(n_samples, n_parameters)
        """

        samples = []
        samples.append(self.var_noise.sample_from_prior(n_samples))
        samples.append(self.mean.sample_from_prior(n_samples))
        samples.append(self.kernel.sample_parameters(n_samples))

        return np.concatenate(samples, 1)

    def sample_parameters_posterior(self, n_samples):
        """

        :param n_samples:
        :return: [np.array(n)]
        """
        parameters = self.get_value_parameters_model
        samples = []
        for j in xrange(n_samples):
            for i in xrange(self.thinning + 1):
                parameters =  self.slice_sampler.slice_sample(parameters)
            samples.append(parameters)
        return samples


    @property
    def get_bounds_parameters(self):
        """
        Get bounds of the parameters of the model.
        :return: [(float, float)]
        """
        bounds = []
        bounds += self.var_noise.bounds
        bounds += self.mean.bounds
        bounds += self.kernel.get_bounds_parameters()

        return bounds

    def mle_parameters(self, start=None):
        """
        Computes the mle parameters of the kernel of a GP process, and mean and variance of the
        noise of the Gaussian regression.

        :param start: (np.array(n)) starting point of the optimization of the llh.

        :return: {
            'solution': (np.array(n)) mle parameters,
            'optimal_value': float,
            'gradient': np.array(n),
            'warnflag': int,
            'task': str
        }
        """

        if start is None:
            start = self.sample_parameters_prior(1)[0, :]

        optimization = Optimization(
            LBFGS_NAME,
            lambda params: self.log_likelihood(params[0], params[1], params[2:]),
            self.get_bounds_parameters,
            lambda params: self.grad_log_likelihood(params[0], params[1], params[2:]),
            minimize=False)

        return optimization.optimize(start)

    def fit_gp_regression(self):
        """
        Fit a GP regression model

        :return: self
        """

        results = self.mle_parameters()

        logger.info("Results of the GP fitting: ")
        logger.info(results)

        self.var_noise.set_value(results['solution'][0])
        self.mean.set_value(results['solution'][1])
        self.kernel.update_value_parameters(results['solution'][2:])

        return self

    def compute_posterior_parameters(self, points):
        """
        Compute the posterior mean and cov of the GP at points:
            f(points) ~ GP(mu_n(points), cov_n(points, points))

        :param points: np.array(nxm)
        :return: {
            'mean': np.array(n),
            'cov': np.array(nxn)
        }
        """
        # TODO: check the case where n > 1

        chol, cov = self._chol_cov_including_noise(
            self.var_noise.value[0], self.kernel.hypers_values_as_array)

        y_unbiased = self.training_data['evaluations'] - self.mean.value[0]
        solve = spla.cho_solve((chol, True), y_unbiased)

        vec_cov = self.kernel.cross_cov(points, self.training_data['points'])

        mu_n = self.mean.value[0] + np.dot(vec_cov, solve)

        solve_2 = spla.cho_solve((chol, True), vec_cov.transpose())
        cov_n = self.kernel.cov(points) - np.dot(vec_cov, solve_2)

        return mu_n, cov_n

    def log_prob_parameters(self, parameters):
        """
        Computes the logarithm of prob(data|parameters)prob(parameters) which is
        proportional to prob(parameters|data).

        :param parameters: (np.array(n)) The order is defined in the function get_parameters_model
            of the class model.
        :param model: (gp_fitting_gaussian) We may include more models in the future.

        :return: float
        """

        lp = 0.0

        parameters_model = self.get_parameters_model

        index = 0
        for parameter in parameters_model:
            dimension = parameter.dimension
            lp += parameter.prior_logprob(parameters[index: dimension])
            index += dimension

        lp += self.log_likelihood(parameters[0], parameters[1], parameters[2:])

        return lp


class GradientGPFittingGaussian(object):

    @staticmethod
    def compute_gradient_llh_given_grad_cov(grad_cov, chol, solve):
        """

        :param grad_cov: np.array(nxn)
        :param chol: (np.array(nxn)) cholesky decomposition of cov
        :param solve: (np.array(n)) cov^-1 (y-mean), where cov = chol * chol^T
        :return: float
        """

        #solve = spla.cho_solve((chol, True), y_unbiased)
        solve = solve.reshape((len(solve), 1))
        solve_1 = spla.cho_solve((chol, True), grad_cov)

        product = np.dot(np.dot(solve, solve.transpose()), grad_cov)

        sol = 0.5 * np.trace(product - solve_1)

        return sol

    @staticmethod
    def compute_gradient_kernel_respect_to_noise(n):
        """
        Computes the gradient of the kernel respect to the variance of the noise.

        :param n: (int) number of training points
        :return: np.array(nxn)
        """

        return np.identity(n)

    @staticmethod
    def compute_gradient_mean(chol, y_unbiased, n):
        """
        Computes the gradient of the llh respect to the mean.

        :param chol: (np.array(nxn)) cholesky decomposition of cov
        :param y_unbiased: (np.array(n)) y_obs - mean
        :param n: (int) number of training points
        :return: float
        """
        solve = spla.cho_solve((chol, True), -1.0 * np.ones(n))
        return -1.0 * np.dot(y_unbiased, solve)


class ValidationGPModel(object):

    @classmethod
    def cross_validation_mle_parameters(cls, type_kernel, training_data, dimensions):
        """
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        """

        n_data = len(training_data['evaluations'])

        noise = True
        if training_data.get('var_noise') is None:
            noise = False


        training_data_sets = {}
        test_points = {}
        gp_objects = {}

        for i in xrange(n_data):
            selector = [x for x in range(n_data) if x != i]
            training_data_sets[i] = {}
            test_points[i] = {}

            training_data_sets[i]['evaluations'] = training_data['evaluations'][selector]
            test_points[i]['evaluations'] = training_data['evaluations'][i]

            training_data_sets[i]['points'] = training_data['points'][selector, :]
            test_points[i]['points'] = training_data['points'][[i], :]

            if noise:
                training_data_sets[i]['var_noise'] = training_data['var_noise'][selector]
                test_points[i]['var_noise'] = training_data['var_noise'][i]

            gp_objects[i] = GPFittingGaussian(type_kernel, training_data_sets[i], dimensions)

        new_gp_objects = Parallel.run_function_different_arguments_parallel(
            wrapper_fit_gp_regression, gp_objects)

        number_correct = 0
        success_runs = 0
        correct = {}
        posterior_parameters = {}

        for i in xrange(n_data):
            if new_gp_objects.get(i) is None:
                logger.info("It wasn't possible to fit the GP for %d"%i)
                continue
            success_runs += 1
            posterior = new_gp_objects[i].compute_posterior_parameters(test_points[i]['points'])
            posterior_parameters[i] = posterior
            if noise:
                correct[i] = cls.check_value_within_ci(
                    test_points[i]['evaluations'], posterior[0], posterior[1][0, 0],
                    var_noise=test_points[i]['var_noise'])
            else:
                correct[i] = cls.check_value_within_ci(
                    test_points[i]['evaluations'], posterior[0], posterior[1][0, 0])
            if correct[i]:
                number_correct += 1

        proportion_success = number_correct / float(success_runs)
        logger.info("Proportion of success is %f"%proportion_success)



        # TODO: ADD PLOTS. Output results to a file.
        plt.errorbar(np.arange(N), means, yerr=2.0 * standard_dev, fmt='o')
        plt.scatter(np.arange(N), y, color='r')
        plt.savefig("diagnostic_kernel.png")

    @staticmethod
    def check_value_within_ci(value, mean, variance, var_noise=None):
        """
        Check if value is within [mean - 1.96 * sqrt(variance), mean + 1.96 * sqrt(variance)]
        :param value: float
        :param mean: float
        :param variance: float
        :param var_noise: float
        :return: booleans
        """

        if var_noise is None:
            var_noise = 0

        std = np.sqrt(variance + var_noise)

        return mean - 2.0 * std <= value <= mean + 2.0 * std