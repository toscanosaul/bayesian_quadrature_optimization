from __future__ import absolute_import

import scipy.linalg as spla
import numpy as np

from os import path

from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    MEAN_NAME,
    VAR_NOISE_NAME,
    CHOL_COV,
    SOL_CHOL_Y_UNBIASED,
    DIAGNOSTIC_KERNEL_DIR,
    SMALLEST_NUMBER,
    SMALLEST_POSITIVE_NUMBER,
    LARGEST_NUMBER,
)
from stratified_bayesian_optimization.lib.util_gp_fitting import (
    get_kernel_default,
    get_kernel_class,
)
from stratified_bayesian_optimization.lib.util import (
    separate_numpy_arrays_in_lists,
    wrapper_fit_gp_regression,
    get_default_values_kernel,
    expand_dimension_vector,
    reduce_dimension_vector,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.lib.constant import LBFGS_NAME
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class GPFittingGaussian(object):

    _possible_kernels_ = [MATERN52_NAME, TASKS_KERNEL_NAME, PRODUCT_KERNELS_SEPARABLE]

    def __init__(self, type_kernel, training_data, dimensions, kernel_values=None, mean_value=None,
                 var_noise_value=None, thinning=0, data=None):
        """
        :param type_kernel: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param training_data: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or [])}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param kernel_values: [float], contains the dafault values of the parameters of the kernel
        :param mean_value: [float], It contains the value of the mean parameter.
        :param var_noise_value: [float], It contains the variance of the noise of the model
        :param thinning: (int)
        :param data: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or [])}, it might contains more points than the points used
            to train the kernel, or different points.
        """

        self.type_kernel = type_kernel
        self.class_kernel = get_kernel_class(type_kernel)

        self.training_data = training_data

        if data is None:
            data = training_data

        self.data = self.convert_from_list_to_numpy(data)

        self.dimensions = dimensions

        if mean_value is None:
            mean_value = [0.0]
        if var_noise_value is None:
            var_noise_value = [SMALLEST_POSITIVE_NUMBER]
        if kernel_values is None:
            kernel_values = get_default_values_kernel(type_kernel, dimensions)

        self.kernel_values = kernel_values
        self.mean_value = mean_value
        self.var_noise_value = var_noise_value

        self.kernel = get_kernel_default(type_kernel, self.dimensions, np.array(kernel_values))

        self.mean = ParameterEntity(
            MEAN_NAME, np.array(mean_value), UniformPrior(1, [SMALLEST_NUMBER], [LARGEST_NUMBER]))
        self.var_noise = ParameterEntity(
            VAR_NOISE_NAME, var_noise_value,
            UniformPrior(1, [SMALLEST_POSITIVE_NUMBER], [LARGEST_NUMBER]))

        self.thinning = thinning
        self.slice_sampler = None

        self.cache_chol_cov = {}
        self.cache_sol_chol_y_unbiased = {}

        self.setUp()

    def setUp(self):
        self.slice_sampler = SliceSampling(self.log_prob_parameters)

    def add_points_evaluations(self, point, evaluation, var_noise_eval=None):
        """

        :param point: np.array(kxm)
        :param evaluation: np.array(k)
        :param var_noise_eval: np.array(k)
        """
        self.data['points'] = np.append(self.data['points'], point, axis=0)
        self.data['evaluation'] = np.append(self.data['evaluation'], evaluation)

        if var_noise_eval is not None:
            self.data['var_noise'] = np.append(self.data['var_noise'], var_noise_eval)

    @staticmethod
    def convert_from_list_to_numpy(data_as_list):
        """
        Conver the lists to numpy arrays.
        :param data_as_list: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or None)}
        :return: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        """

        data = {}
        n_points = len(data_as_list['points'])
        dim_point = len(data_as_list['points'][0])

        points = np.zeros((n_points, dim_point))
        evaluations = np.zeros(n_points)
        var_noise = None

        if len(data_as_list['var_noise']) > 0:
            var_noise = np.zeros(n_points)
            iterate = zip(data_as_list['points'], data_as_list['evaluations'],
                          data_as_list['var_noise'])
        else:
            iterate = zip(data_as_list['points'], data_as_list['evaluations'])


        for index, point in enumerate(iterate):
            points[index, :] = point[0]
            evaluations[index] = point[1]
            if len(data_as_list['var_noise']) > 0:
                var_noise[index] = point[2]

        data['points'] = points
        data['evaluations'] = evaluations
        data['var_noise'] = var_noise

        return data

    @staticmethod
    def convert_from_numpy_to_list(data_as_np):
        """
         Conver the numpy arrays to lists.
         :param data_as_np: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
         :return: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
             'var_noise': ([float],dim=n or None)}
         """
        evaluations = list(data_as_np['evaluations'])

        if data_as_np['var_noise'] is not None:
            var_noise = list(data_as_np['var_noise'])
        else:
            var_noise = []

        n_points = len(evaluations)
        points = [list(data_as_np['points'][i, :]) for i in xrange(n_points)]

        data = {}
        data['points'] = points
        data['var_noise'] = var_noise
        data['evaluations'] = evaluations
        return data

    def serialize(self):
        return {
            'type_kernel': self.type_kernel,
            'training_data': self.training_data,
            'dimensions': self.dimensions,
            'kernel_values': self.kernel_values,
            'mean_value': self.mean_value,
            'var_noise_value': self.var_noise_value,
            'thinning': self.thinning,
            'data': self.convert_from_numpy_to_list(self.data)
        }

    @classmethod
    def deserialize(cls, s):
        return cls(**s)

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

        n = self.data['points'].shape[0]

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.dimensions[0]),
                separate_numpy_arrays_in_lists(self.data['points'], self.dimensions[0]),
                self.dimensions, self.type_kernel[1: ])
        else:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                parameters_kernel, self.data['points'], self.dimensions[0]
            )

        if self.data.get('var_noise'):
            cov += np.diag(self.data['var_noise'])

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
        y_unbiased = self.data['evaluations'] - mean

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
                separate_numpy_arrays_in_lists(self.data['points'], self.dimensions[0]),
                self.dimensions, self.type_kernel[1: ])
        else:
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                parameters_kernel, self.data['points'], self.dimensions[0])

        chol, cov = self._chol_cov_including_noise(var_noise, parameters_kernel)

        y_unbiased = self.data['evaluations'] - mean

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

        n_training_points = self.data.shape[0]

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

    def sample_parameters_prior(self, n_samples, random_seed=None):
        """
        Sample parameters of the GP model from their prior

        :param n_samples: int
        :param random_seed: int
        :return: np.array(n_samples, n_parameters)
        """

        if random_seed is not None:
            np.random_seed(random_seed)
        samples = []
        samples.append(self.var_noise.sample_from_prior(n_samples))
        samples.append(self.mean.sample_from_prior(n_samples))
        samples.append(self.kernel.sample_parameters(n_samples))

        return np.concatenate(samples, 1)

    def sample_parameters_posterior(self, n_samples, random_seed=None):
        """
        Sample parameters of the GP model from their posterior.

        :param n_samples:
        :param random_seed: (int)
        :return: np.array(n_samples, n_parameters)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        parameters = self.get_value_parameters_model
        samples = []
        for j in xrange(n_samples):
            for i in xrange(self.thinning + 1):
                parameters =  self.slice_sampler.slice_sample(parameters)
            samples.append(parameters)
        return np.concatenate(samples, 1)

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

    def mle_parameters(self, start=None, indexes=None):
        """
        Computes the mle parameters of the kernel of a GP process, and mean and variance of the
        noise of the Gaussian regression.

        :param start: (np.array(n)) starting point of the optimization of the llh.
        :parameter indexes: [int], we optimize the MLE only over all the parameters, but the
            parameters of the indexes. If it's None, we optimize over all the parameters.

        :return: {
            'solution': (np.array(n)) mle parameters,
            'optimal_value': float,
            'gradient': np.array(n),
            'warnflag': int,
            'task': str
        }
        """

        if start is None:
            start = self.sample_parameters_posterior(1)[0, :]

        def default_objective_function(params):
            return self.log_likelihood(params[0], params[1], params[2:])

        def default_grad_function(params):
            return self.grad_log_likelihood(params[0], params[1], params[2:])

        bounds = self.get_bounds_parameters

        if indexes is not None:
            default_values = self.get_value_parameters_model
            change_indexes_ = [i for i in xrange(len(default_values)) if i not in indexes]

            default_bounds = list(self.get_bounds_parameters)
            bounds = []
            for j in change_indexes_:
                bounds.append(default_bounds[j])

            def objective_function(params):
                new_params = expand_dimension_vector(
                    params, change_indexes_, default_values)
                return default_objective_function(new_params)

            def grad_function(params):
                new_params = expand_dimension_vector(
                    params, change_indexes_, default_values)
                grad = default_grad_function(new_params)
                new_grad = reduce_dimension_vector(grad, change_indexes_)
                return new_grad
        else:
            objective_function = default_objective_function
            grad_function = default_grad_function

        optimization = Optimization(
            LBFGS_NAME,
            objective_function,
            bounds,
            grad_function,
            minimize=False)

        return optimization.optimize(start)

    def fit_gp_regression(self, indexes=None):
        """
        Fit a GP regression model

        :parameter indexes: [int],  we optimize the MLE only over all the parameters, but the
            parameters of the indexes. If it's None, we optimize over all the parameters.
        :return: self
        """

        results = self.mle_parameters(indexes)

        logger.info("Results of the GP fitting: ")
        logger.info(results)

        self.var_noise.set_value(results['solution'][0])
        self.mean.set_value(results['solution'][1])
        self.kernel.update_value_parameters(results['solution'][2:])

        self.kernel_values = results['solution'][2:]
        self.mean_value = results['solution'][1]
        self.var_noise_value = results['solution'][0]

        return self

    @classmethod
    def train(cls, type_kernel, dimensions, mle, training_data, thinning=0):
        """
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param mle: (boolean) If true, fits the GP by MLE.
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}.
        :param thinning: (int)
        :return: GPFittingGaussian
        """

        if mle:
            gp = cls(type_kernel, training_data, dimensions, thinning)
            return gp.fit_gp_regression()

        return cls(type_kernel, training_data, dimensions, thinning)

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

        y_unbiased = self.data['evaluations'] - self.mean.value[0]
        solve = spla.cho_solve((chol, True), y_unbiased)

        vec_cov = self.kernel.cross_cov(points, self.data['points'])

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
    _validation_filename = 'validation_kernel_{problem}.json'.format
    _validation_filename_plot = 'validation_kernel_mean_vs_observations_{problem}.png'.format
    _validation_filename_histogram = 'validation_kernel_histogram_{problem}.png'.format

    @classmethod
    def cross_validation_mle_parameters(cls, type_kernel, training_data, dimensions, problem_name):
        """
        A json file with the percentage of success is generated. The output can be used to create
        a histogram and a diagnostic plot.

        The histogram would be of the vector (y_eval-means)/std_vec. We'd expect to have an
        histogram similar to the one of a standard Gaussian random variable.

         Diagnostic plot: For each point of the test fold, we plot the value of the function in that
         point, and its C.I. based on the GP model. We would expect that around 95% of the test
         points stay within their C.I.

        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param problem_name: (str)

        :return: {
            'means': (np.array(n)), vector with the means of the GP at the points chosen in each of
                the test folds.
            'std_vec': (np.array(n)), vector with the std of the GP at the points chosen in each of
                the test folds.
            'y_eval': np.array(n), vector with the actual values of the function at the chosen
                points.
            'n_data': int, total number of points.
            'filename_plot': str,
            'filename_histogram': str
        }
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

        means = np.zeros(n_data)
        std_vec = np.zeros(n_data)
        y_eval = np.zeros(n_data)

        for i in xrange(n_data):
            if new_gp_objects.get(i) is None:
                logger.info("It wasn't possible to fit the GP for %d"%i)
                continue
            success_runs += 1
            posterior = new_gp_objects[i].compute_posterior_parameters(test_points[i]['points'])
            posterior_parameters[i] = posterior

            means[i] = posterior[0]
            std_vec[i] = np.sqrt(posterior[1][0, 0])
            y_eval[i] = test_points[i]['evaluations']

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

        filename = path.join(DIAGNOSTIC_KERNEL_DIR, cls._validation_filename(
            problem=problem_name
        ))

        JSONFile.write({'Percentage of success: ': proportion_success}, filename)


        filename_plot = path.join(DIAGNOSTIC_KERNEL_DIR, cls._validation_filename_plot(
            problem=problem_name
        ))

        filename_histogram = path.join(DIAGNOSTIC_KERNEL_DIR, cls._validation_filename_histogram(
            problem=problem_name
        ))

        return {
            'means': means,
            'std_vec': std_vec,
            'y_eval': y_eval,
            'n_data': n_data,
            'filename_plot': filename_plot,
            'filename_histogram': filename_histogram,
        }

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