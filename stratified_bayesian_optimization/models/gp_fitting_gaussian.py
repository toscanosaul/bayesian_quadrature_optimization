from __future__ import absolute_import

from os import path
import os
import sys

from numpy.linalg.linalg import LinAlgError
import numpy as np

from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    ORNSTEIN_KERNEL,
    PRODUCT_KERNELS_SEPARABLE,
    MEAN_NAME,
    VAR_NOISE_NAME,
    CHOL_COV,
    SOL_CHOL_Y_UNBIASED,
    DIAGNOSTIC_KERNEL_DIR,
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
    LENGTH_SCALE_NAME,
    SMALLEST_POSITIVE_NUMBER,
    SCALED_KERNEL,
    LBFGS_NAME,
    SAME_CORRELATION,
    RESULTS_DIR,
    EI_METHOD,
    SGD_NAME,
    DEBUGGING_DIR,
    DEFAULT_N_PARAMETERS,
)
from stratified_bayesian_optimization.lib.util_gp_fitting import (
    get_kernel_default,
    get_kernel_class,
    parameters_kernel_from_list_to_dict,
    wrapper_log_prob,
    define_prior_parameters_using_data,
)
from stratified_bayesian_optimization.lib.util import (
    separate_numpy_arrays_in_lists,
    wrapper_fit_gp_regression,
    get_default_values_kernel,
    get_number_parameters_kernel,
    combine_vectors,
    separate_vector,
    wrapper_GPFittingGaussian,
    wrapper_objective_acquisition_function,
    wrapper_gradient_posterior_mean_gp_model,
    wrapper_posterior_mean_gp_model,
    wrapper_optimize,
    wrapper_sgd,
    wrapper_evaluate_gradient_sample_params_gp,
)
from stratified_bayesian_optimization.services.domain import (
    DomainService,
)
from stratified_bayesian_optimization.lib.optimization import Optimization
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.priors.non_negative import NonNegativePrior
from stratified_bayesian_optimization.priors.horseshoe import HorseShoePrior
from stratified_bayesian_optimization.priors.gaussian import GaussianPrior
from stratified_bayesian_optimization.priors.constant import Constant
from stratified_bayesian_optimization.samplers.slice_sampling import SliceSampling
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.la_functions import (
    cholesky,
    cho_solve,
)

logger = SBOLog(__name__)


class GPFittingGaussian(object):

    _possible_kernels_ = [MATERN52_NAME, TASKS_KERNEL_NAME, PRODUCT_KERNELS_SEPARABLE]

    def __init__(self, type_kernel, training_data, dimensions=None, bounds_domain=None,
                 kernel_values=None, mean_value=None, var_noise_value=None, thinning=0, n_burning=0,
                 start_point_sampler=None, max_steps_out=1, data=None, random_seed=None,
                 type_bounds=None, training_name=None, problem_name=None,
                 name_model='gp_fitting_gaussian', samples_parameters=None, noise=False,
                 simplex_domain=None, **kernel_parameters):
        """
        :param type_kernel: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL].
            If we want to use a scaled NAME_1_KERNEL, the parameter must be
            [SCALED_KERNEL, NAME_1_KERNEL].
        :param training_data: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or [])}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
        :param bounds_domain: [[float, float]], lower bound and upper bound for each entry. This
            parameter is used to compute priors in a smart way.
        :param kernel_values: [float], contains the default values of the parameters of the kernel
        :param mean_value: [float], It contains the value of the mean parameter.
        :param var_noise_value: [float], It contains the variance of the noise of the model
        :param thinning: (int) Parameters of the MCMC. If it's 1, we take all the samples.
        :param n_burning: (int) Number of burnings samples for the MCMC.
        :param start_point_sampler: [float], starting point for the MCMC sampler (we won't take
            burning samples anymore).
        :param max_steps_out: (int) Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param data: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or [])}, it might contains more points than the points used
            to train the kernel, or different points. If its None, it's replaced by the
            traning_data.
        :param random_seed: int
        :param type_bounds:  [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param training_name: (str)
        :param problem_name: (str)
        :param name_model: (str)
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
        :param samples_parameters: [[float]]

        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if type_bounds is None and bounds_domain is not None:
            type_bounds = len(bounds_domain) * [0]

        if type_bounds is None and bounds_domain is None:
            type_bounds = []

        self.name_model = name_model
        self.training_name = training_name
        self.problem_name = problem_name

        self.noise = noise

        self.type_kernel = type_kernel

        self.additional_kernel_parameters = kernel_parameters
        self.simplex_domain = simplex_domain

        self.class_kernel = get_kernel_class(type_kernel[0])
        self.bounds = bounds_domain
        self.training_data = training_data
        self.training_data_as_array = self.convert_from_list_to_numpy(training_data)
     #   self.dimension_domain = self.training_data_as_array['points'].shape[1]
        self.dimension_domain = len(bounds_domain)
        self.dimensions = dimensions
        self.type_bounds = type_bounds

        if data is None:
            data = training_data
        self.data = self.convert_from_list_to_numpy(data)

        if mean_value == []:
            mean_value = None

        if kernel_values == []:
            kernel_values = None

        if var_noise_value == []:
            var_noise_value = None

        self.kernel_values = kernel_values
        self.mean_value = mean_value
        self.var_noise_value = var_noise_value

        self.kernel = None
        self.mean = None
        self.var_noise = None
        self.kernel_dimensions = None  # Only used for the PRODUCT_KERNELS_SEPARABLE.

        # Number of parameters of only the kernel. Only used for the PRODUCT_KERNELS_SEPARABLE.
        self.number_parameters = None

        self.length_scale_indexes = None  # Indexes of the length scale parameter

        self.dimension_parameters = None  # Total number of parameters

        self.thinning = thinning
        self.max_steps_out = max_steps_out
        self.n_burning = n_burning
        self.samples_parameters = []

        if samples_parameters is not None:
            self.samples_parameters = [np.array(sample) for sample in samples_parameters]
        self.slice_samplers = []
        self.start_point_sampler = start_point_sampler

        self.cache_chol_cov = {}
        self.cache_sol_chol_y_unbiased = {}

        self.best_solution = {} # Historical best solution for EI.
        self.cache_cov_n = {} # Cache computations of the cov_n

        self.optimization_results = []

        self.set_parameters_kernel()
        self.set_samplers()

        if type_bounds is not None and len(type_bounds) > 0 and type_bounds[-1] == 1:
            self.separate_tasks = True
            self.tasks = self.bounds[-1]
        else:
            self.separate_tasks = False
        self.model_only_x = False

    def set_samplers(self):
        """
        Defines the samplers of the parameters of the model.
        We assume that we only have one set of length scale parameters.
        """

        if self.length_scale_indexes is None:
            self.slice_samplers.append(SliceSampling(wrapper_log_prob,
                                                     range(self.dimension_parameters)))
        else:
            slice_parameters = {
                'max_steps_out': self.max_steps_out,
                'component_wise': False,
            }
            indexes = [i for i in range(self.dimension_parameters) if i not in
                       self.length_scale_indexes]
            ignore_index = None
            if not self.noise or self.data.get('var_noise') is not None:
                ignore_index = [0, 1]

            if ORNSTEIN_KERNEL in self.type_kernel:
                if ignore_index is None:
                    ignore_index = []
                ignore_index += [2]

            if len(indexes) != len(ignore_index):
                self.slice_samplers.append(
                    SliceSampling(
                        wrapper_log_prob, indexes, ignore_index=ignore_index, **slice_parameters))

            slice_parameters['component_wise'] = True
            self.slice_samplers.append(SliceSampling(wrapper_log_prob, self.length_scale_indexes,
                                                     **slice_parameters))

        if self.start_point_sampler is not None and len(self.start_point_sampler) > 0:
            if len(self.samples_parameters) == 0:
                self.samples_parameters.append(np.array(self.start_point_sampler))
        else:
            self.samples_parameters = []
            self.samples_parameters.append(self.get_value_parameters_model)
            if self.n_burning > 0:
                parameters = self.sample_parameters(float(self.n_burning) / (self.thinning + 1))
                self.samples_parameters = []
                self.samples_parameters.append(parameters[-1])
                self.start_point_sampler = parameters[-1]
            else:
                self.start_point_sampler = self.get_value_parameters_model

    def start_new_chain(self, random_seed=None):
        """
        Starts a new chain of sampled parameters.

        :param random_seed: int
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if self.n_burning > 0:
            parameters = self.sample_parameters(float(self.n_burning) / (self.thinning + 1))
        else:
            parameters = [self.samples_parameters[-1]]

        self.samples_parameters = []
        self.samples_parameters.append(parameters[-1])
        self.start_point_sampler = parameters[-1]

    def sample_parameters(self, n_samples, start_point=None, random_seed=None):
        """
        Sample parameters of the model from the posterior without considering burning.

        :param n_samples: (int)
        :param start_point: np.array(n_parameters)
        :param random_seed: int

        :return: n_samples * [np.array(float)]
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        samples = []

        if start_point is None:
            start_point = self.samples_parameters[-1]


        n_samples *= (self.thinning + 1)
        n_samples = int(n_samples)

        if len(self.slice_samplers) == 1:
            for sample in xrange(n_samples):
                start_point_ = None
                n_try = 0
                points = separate_vector(start_point, self.length_scale_indexes)
                while start_point_ is None and n_try < 10:
                    try:
                        start_point_ = \
                            self.slice_samplers[0].slice_sample(points[0], points[1], *(self, ))
                    except Exception as e:
                        n_try += 1
                        start_point_ = None
                if start_point_ is None:
                    logger.info('program failed to compute a sample of the parameters')
                    sys.exit(1)
                start_point = combine_vectors(start_point_, points[1], self.length_scale_indexes)
               # start_point = start_point_
                samples.append(start_point)
            return samples[::self.thinning + 1]

        for sample in xrange(n_samples):
            print "sample"
            print sample
            print n_samples
            points = separate_vector(start_point, self.length_scale_indexes)
            for index, slice in enumerate(self.slice_samplers):
                new_point_ = None
                n_try = 0
                while new_point_ is None and n_try < 10:
                    try:
                        new_point_ = \
                            slice.slice_sample(points[1 - index], points[index], *(self, ))
                    except Exception as e:
                        n_try += 1
                        new_point_ = None
                if new_point_ is None:
                    logger.info('program failed to compute a sample of the parameters')
                    sys.exit(1)
                points[1 - index] = new_point_
            start_point = combine_vectors(points[0], points[1], self.length_scale_indexes)
            samples.append(start_point)
        samples_return = samples[::self.thinning + 1]
        self.samples_parameters += samples_return

        return samples_return

    def set_parameters_kernel(self):
        """
        Defines the mean and var_noise parameters. It also defines the kernel.
        """

        prior_parameters_values = self.get_values_parameters_from_data(
            self.kernel_values, self.mean_value, self.var_noise_value, self.type_kernel,
            self.dimensions, **self.additional_kernel_parameters)

        if not self.noise or self.data.get('var_noise') is not None:
            self.mean_value = [0.0]
            self.var_noise_value = [0.0]

        parameters_priors = prior_parameters_values['kernel_values']

        parameters_priors = parameters_kernel_from_list_to_dict(parameters_priors, self.type_kernel,
                                                                self.dimensions)

        if self.kernel_values is None:
            self.kernel_values = list(
                get_default_values_kernel(self.type_kernel, self.dimensions, **parameters_priors))

        if self.mean_value is None:
            self.mean_value = list(prior_parameters_values['mean_value'])

        if self.var_noise_value is None:
            self.var_noise_value = list(prior_parameters_values['var_noise_value'])

        if self.noise and self.data.get('var_noise') is None:
            self.mean = ParameterEntity(
                MEAN_NAME, np.array(self.mean_value), GaussianPrior(1, self.mean_value[0], 1.0))
        else:
            self.mean = ParameterEntity(
                MEAN_NAME, np.array([0.0]), Constant(1, 0.0))

        if self.noise and self.data.get('var_noise') is None:
            self.var_noise = ParameterEntity(
                VAR_NOISE_NAME, np.array(self.var_noise_value),
                NonNegativePrior(1, HorseShoePrior(1, self.var_noise_value[0])),
                bounds=[(SMALLEST_POSITIVE_NUMBER, None)])
        else:
            self.var_noise = ParameterEntity(
                VAR_NOISE_NAME, np.array([0.0]),
                Constant(1, 0.0),
                bounds=[(0.0, 0.0)])

        self.kernel = get_kernel_default(self.type_kernel, self.dimensions, self.bounds,
                                         np.array(self.kernel_values), parameters_priors,
                                         **self.additional_kernel_parameters)

        self.dimension_parameters = self.kernel.dimension_parameters + 2

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            self.kernel_dimensions = [self.kernel.dimension]
            if len(self.type_kernel) > 1:
                for name in self.kernel.names:
                    self.kernel_dimensions.append(self.kernel.kernels[name].dimension)

            # I think that this is only useful for the product of kernels.
            self.number_parameters = [get_number_parameters_kernel(
                self.type_kernel, self.dimensions, **self.additional_kernel_parameters)]
            if len(self.dimensions) > 1:
                for type_k, dim in zip(self.type_kernel[1:], self.dimensions[1:]):
                    self.number_parameters.append(
                        get_number_parameters_kernel([type_k], [dim],
                                                     **self.additional_kernel_parameters))

        self.length_scale_indexes = self.get_indexes_length_scale()

    def get_indexes_length_scale(self):
        """
        Get indexes of the length scale parameters

        :return: [int]
        """
        length_scale_indexes = None
        parameters = self.kernel.hypers_as_list
        index = 2
        for parameter in parameters:
            if parameter.name == LENGTH_SCALE_NAME:
                if length_scale_indexes is None:
                    length_scale_indexes = []
                length_scale_indexes += list(np.arange(parameter.dimension) + index)
                continue
            index += parameter.dimension
        return length_scale_indexes

    def get_values_parameters_from_data(self, kernel_values, mean_value, var_noise_value,
                                        type_kernel, dimensions, **kernel_parameters):
        """
        Defines value of the parameters of the prior distributions of the model's parameters.

        :param kernel_values: [float], contains the default values of the parameters of the kernel
        :param mean_value: [float], It contains the value of the mean parameter.
        :param var_noise_value: [float], It contains the variance of the noise of the model
        :param type_kernel: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
        :return: {
            'kernel_values': [float],
            'mean_value': [float],
            'var_noise_value': [float],
        }
        """
        if mean_value is None:
            mu = np.mean(self.training_data_as_array['evaluations'])
            mean_value = [mu]

        if var_noise_value is None:
            if len(self.training_data_as_array['evaluations']) > 1:
                var_evaluations = np.var(self.training_data_as_array['evaluations'])
                var_noise_value = [var_evaluations]
            else:
                var_noise_value = [0.1]

        if kernel_values is None:
            kernel_parameters_values = define_prior_parameters_using_data(
                self.training_data_as_array,
                type_kernel,
                dimensions,
                sigma2=var_noise_value[0],
                **kernel_parameters
            )

            kernel_values = get_default_values_kernel(type_kernel, dimensions,
                                                      **kernel_parameters_values)

        return {
            'kernel_values': kernel_values,
            'mean_value': mean_value,
            'var_noise_value': var_noise_value,
        }

    def add_points_evaluations(self, point, evaluation, var_noise_eval=None):
        """

        :param point: np.array(kxm)
        :param evaluation: np.array(k)
        :param var_noise_eval: np.array(k)
        """

        self.data['points'] = np.append(self.data['points'], point, axis=0)
        self.data['evaluations'] = np.append(self.data['evaluations'], evaluation)

        if var_noise_eval is not None:
            self.data['var_noise'] = np.append(self.data['var_noise'], var_noise_eval)

        # TODO: updated cache in a smart way
        # https://math.stackexchange.com/questions/955874/cholesky-factor-when-adding-a-row-and-
        # column-to-already-factorized-matrix

        self.cache_chol_cov = {}
        self.cache_sol_chol_y_unbiased = {}

    @staticmethod
    def convert_from_list_to_numpy(data_as_list):
        """
        Conver the lists to numpy arrays.
        :param data_as_list: {'points': ([[float]], dim=nxm), 'evaluations': ([float],dim=n),
            'var_noise': ([float],dim=n or None)}
        :return: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        """

        if len(data_as_list['points']) == 0:
            return {'points': None, 'evaluations': None, 'var_noise': None}

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
        bounds = self.bounds
        if self.bounds is None:
            bounds = []

        training_name = self.training_name
        if self.training_name is None:
            training_name = ''

        problem_name = self.problem_name
        if self.problem_name is None:
            problem_name = ''

        same_correlation = self.additional_kernel_parameters.get(SAME_CORRELATION, False)

        samples_parameters = self.samples_parameters
        if len(samples_parameters) > 0:
            samples_parameters = [list(param) for param in samples_parameters]

        return {
            'type_kernel': self.type_kernel,
            'training_data': self.training_data,
            'dimensions': self.dimensions,
            'kernel_values': list(self.kernel_values),
            'mean_value': list(self.mean_value),
            'var_noise_value': list(self.var_noise_value),
            'thinning': self.thinning,
            'n_burning': self.n_burning,
            'max_steps_out': self.max_steps_out,
            'data': self.convert_from_numpy_to_list(self.data),
            'bounds_domain': bounds,
            'type_bounds': self.type_bounds,
            'name_model': self.name_model,
            'training_name': training_name,
            'problem_name': problem_name,
            'same_correlation': same_correlation,
            'start_point_sampler': list(self.start_point_sampler),
            'samples_parameters': samples_parameters,
        }

    @classmethod
    def deserialize(cls, s, use_only_training_points=True):
        """

        :param s:
        :param use_only_training_points (boolean) If true,
            it uses only the training points in data. Otherwise, it also includes new points
            previously computed.
        :return: gp-model instance
        """

        model = cls(**s)
        if use_only_training_points:
            model.data = model.convert_from_list_to_numpy(model.training_data)
        return model

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

    def _get_cached_data(self, index, name, cache=True):
        """

        :param index: tuple associated to the type.
            -(var_noise, parameters_kernel) if name is CHOL_COV
            -(var_noise, parameters_kernel, mean) if name is SOL_CHOL_Y_UNBIASED
        :param name: (str) SOL_CHOL_Y_UNBIASED or CHOL_COV
        :param cache : (boolean) get cached data only if cache is True
        :return: cached data if it's cached, otherwise False
            -(chol, cov) if name is CHOL_COV
            -cov^-1 (y-mean) if name is SOL_CHOL_Y_UNBIASED
        """

        if cache is False:
            return False

        if name == CHOL_COV:
            if index in self.cache_chol_cov:
                return self.cache_chol_cov[index]
        if name == SOL_CHOL_Y_UNBIASED:
            if index in self.cache_sol_chol_y_unbiased:
                return self.cache_sol_chol_y_unbiased[index]
        return False

    def _updated_cached_data(self, index, value, name, clear_cache=True):
        """

        :param index: tuple associated to the type.
            -(var_noise, parameters_kernel) if CHOL_COV
            -(var_noise, parameters_kernel, mean) if SOL_CHOL_Y_UNBIASED
        :param value: value to be cached
        :param name: (str) SOL_CHOL_Y_UNBIASED or CHOL_COV

        """
        if name == CHOL_COV:
            if clear_cache:
                self.cache_chol_cov = {}
                self.cache_sol_chol_y_unbiased = {}
            self.cache_chol_cov[index] = value
        if name == SOL_CHOL_Y_UNBIASED:
            if clear_cache:
                self.cache_sol_chol_y_unbiased = {}
            self.cache_sol_chol_y_unbiased[index] = value

    def evaluate_cov(self, points, parameters_kernel):
        """
        Evaluate the covariance of the kernel of the model on the points.

        :param points: np.array(nxk)
        :param parameters_kernel: np.array(l)

        :return: np.array(nxn)
        """

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            inputs = separate_numpy_arrays_in_lists(points, self.kernel_dimensions[1])
            inputs_dict = {}
            for index, input in enumerate(inputs):
                inputs_dict[self.type_kernel[index + 1]] = input

            cov = self.class_kernel.evaluate_cov_defined_by_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.number_parameters[1]),
                inputs_dict,
                self.dimensions[1:], self.type_kernel[1:], **self.additional_kernel_parameters)
        elif self.type_kernel[0] == SCALED_KERNEL:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                parameters_kernel, points, self.dimensions[0],
                *([self.type_kernel[1]],)
            )
        else:
            cov = self.class_kernel.evaluate_cov_defined_by_params(
                parameters_kernel, points, self.dimensions[0], **self.additional_kernel_parameters
            )

        return cov

    def _chol_cov_including_noise(self, var_noise, parameters_kernel, historical_points=None,
                                  cache=True, clear_cache=True):
        """
        Compute the Cholesky decomposition of
        covariance = cov_kernel + np.diag(var_noise_observations) + np.diag(var_noise), and the
        covariance matrix
        :param var_noise: float
        :param parameters_kernel: np.array(k)
        :param historical_points: np.array(nxk)
        :param cache: (boolean) get cached data only if cache is True
        :return: np.array(nxn) (chol), np.array(nxn) (cov)
        """

        if historical_points is None:
            historical_points = self.data['points']

        cached = self._get_cached_data((var_noise, tuple(parameters_kernel)), CHOL_COV, cache=cache)
        if cached is not False:
            return cached

        n = historical_points.shape[0]

        cov = self.evaluate_cov(historical_points, parameters_kernel)

        if self.data.get('var_noise') is not None:
            cov += np.diag(self.data['var_noise'])

        cov += np.diag(var_noise * np.ones(n))

        chol = cholesky(cov,  max_tries=7)

        if cache:
            self._updated_cached_data((var_noise, tuple(parameters_kernel)), (chol, cov), CHOL_COV,
                                      clear_cache=clear_cache)

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

        cached_solve = self._get_cached_data((var_noise, tuple(parameters_kernel), mean),
                                             SOL_CHOL_Y_UNBIASED)

        if cached_solve is False:
            solve = cho_solve(chol, y_unbiased)
            self._updated_cached_data((var_noise, tuple(parameters_kernel), mean), solve,
                                      SOL_CHOL_Y_UNBIASED)
        else:
            solve = cached_solve

        return -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(y_unbiased, solve)

    def evaluate_grad_cov(self, parameters_kernel, points):
        """
        Evaluate the gradient of the covariance using kernel of the model on the points.

        :param points: np.array(nxk)
        :param parameters_kernel: np.array(l)

        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            inputs = separate_numpy_arrays_in_lists(points, self.kernel_dimensions[1])
            inputs_dict = {}
            for index, input in enumerate(inputs):
                inputs_dict[self.type_kernel[index + 1]] = input
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.number_parameters[1]),
                inputs_dict,
                self.dimensions[1:], self.type_kernel[1:], **self.additional_kernel_parameters)
        elif self.type_kernel[0] == SCALED_KERNEL:
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                parameters_kernel, points, self.dimensions[0],
                *([self.type_kernel[1]],))
        else:
            grad_cov = self.class_kernel.evaluate_grad_defined_by_params_respect_params(
                parameters_kernel, points, self.dimensions[0], **self.additional_kernel_parameters)

        return grad_cov


    def grad_log_likelihood_dict(self, var_noise, mean, parameters_kernel):
        """
        Computes the gradient of the log likelihood

        :param var_noise: (float) variance of the noise
        :param mean: (float)
        :param parameters_kernel: np.array(k), The order of the parameters is given in the
            definition of the class kernel.
        :return: {'var_noise': float, 'mean': float, 'kernel_params': np.array(n)}
        """

        grad_cov = self.evaluate_grad_cov(parameters_kernel, self.data['points'])

        chol, cov = self._chol_cov_including_noise(var_noise, parameters_kernel)

        y_unbiased = self.data['evaluations'] - mean

        cached_solve = self._get_cached_data((var_noise, tuple(parameters_kernel), mean),
                                             SOL_CHOL_Y_UNBIASED)
        if cached_solve is False:
            solve = cho_solve(chol, y_unbiased)
            self._updated_cached_data((var_noise, tuple(parameters_kernel), mean), solve,
                                      SOL_CHOL_Y_UNBIASED)
        else:
            solve = cached_solve

        gradient_kernel_params = np.zeros(len(parameters_kernel))
        for i in xrange(len(parameters_kernel)):
            gradient_kernel_params[i] = GradientGPFittingGaussian.\
                compute_gradient_llh_given_grad_cov(grad_cov[i], chol, solve)

        n_training_points = self.data['points'].shape[0]

        gradient = {}
        gradient['kernel_params'] = gradient_kernel_params
        gradient['mean'] = GradientGPFittingGaussian.compute_gradient_mean(
            chol, y_unbiased, n_training_points)

        grad_var_noise = GradientGPFittingGaussian.compute_gradient_kernel_respect_to_noise(
            n_training_points)
        gradient['var_noise'] = GradientGPFittingGaussian.compute_gradient_llh_given_grad_cov(
            grad_var_noise, chol, solve)

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
            np.random.seed(random_seed)
        samples = []
        samples.append(self.var_noise.sample_from_prior(n_samples))
        samples.append(self.mean.sample_from_prior(n_samples))
        samples.append(self.kernel.sample_parameters(n_samples))

        return np.concatenate(samples, 1)

    def sample_parameters_posterior(self, n_samples, random_seed=None, start_point=None):
        """
        Sample parameters of the GP model from their posterior.

        :param n_samples: int
        :param random_seed: (int)
        :param start_point: np.array(n_parameters)
        :return: np.array(n_samples, n_parameters)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        samples = self.sample_parameters(n_samples, start_point=start_point)

        return np.concatenate(samples).reshape(len(samples), len(samples[0]))

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

    def objective_llh(self, params):
        """
        Function optimized in mle_parameters.
        The function tries to evaluate the objective, if it's not possible,
        it returns minus infinity.

        :param params: np.array(n)
        :return: float
        """

        try:
            obj = self.log_likelihood(params[0], params[1], params[2:])
        except (LinAlgError, ZeroDivisionError, ValueError):
            obj = -np.inf
        return obj

    def grad_llh(self, params):
        """
        Gradient of objective_llh.
        If one of its entries is infinity, the function rounds it using np.clip.

        :param params: np.array(n)
        :return: np.array(n)
        """
        grad = np.clip(self.grad_log_likelihood(params[0], params[1], params[2:]), SMALLEST_NUMBER,
                       LARGEST_NUMBER)

        return grad

    def mle_parameters(self, start=None, random_seed=None):
        """
        Computes the mle parameters of the kernel of a GP process, and mean and variance of the
        noise of the Gaussian regression.

        :param start: (np.array(n)) starting point of the optimization of the llh. If indexes
            is not none, start shouldn't contain the values of the entries in those indexes.
        :param random_seed: int

        :return: {
            'solution': (np.array(n)) mle parameters,
            'optimal_value': float,
            'gradient': np.array(n),
            'warnflag': int,
            'task': str
        }
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        if start is None:
            start = self.sample_parameters_posterior(1)[0, :]

        bounds = self.get_bounds_parameters

        objective_function = self.objective_llh
        grad_function = self.grad_llh

        optimization = Optimization(
            LBFGS_NAME,
            objective_function,
            bounds,
            grad_function,
            minimize=False)

        return optimization.optimize(start)

    def fit_gp_regression(self, start=None, random_seed=None):
        """
        Fit a GP regression model

        :parameter start: (np.array(n)) starting point of the optimization of the llh.
        :parameter random_seed: int

        :return: self
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        results = self.mle_parameters(start=start)

        logger.info("Results of the GP fitting: ")
        logger.info(results)

        sol = results['solution']

        self.update_value_parameters(sol)

        return self

    def update_value_parameters(self, vector):
        """
        Update values of the parameters of the model.

        :param vector: np.array(n)
        """

        self.var_noise.set_value(np.array([vector[0]]))
        self.mean.set_value(np.array([vector[1]]))
        self.kernel.update_value_parameters(vector[2:])

        self.kernel_values = vector[2:]
        self.mean_value = vector[1:2]
        self.var_noise_value = vector[0:1]

    @classmethod
    def train(cls, type_kernel, dimensions, mle, training_data, bounds_domain, thinning=0,
              n_burning=0, max_steps_out=1, random_seed=None, type_bounds=None, training_name=None,
              problem_name=None, kernel_values=None, mean_value=None, var_noise_value=None,
              same_correlation=False, simplex_domain=None):
        """
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param mle: (boolean) If true, fits the GP by MLE.
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}.
        :param bounds_domain: [[float, float]], lower bound and upper bound for each entry. This
            parameter is to compute priors in a smart way.
        :param thinning: (int)
        :param n_burning: (int) Number of burnings samples for the MCMC.
        :param max_steps_out: (int) Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param random_seed: int
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param training_name: (str)
        :parma problem_name: (str)
        :param kernel_values: [float], contains the default values of the parameters of the kernel
        :param mean_value: [float], It contains the value of the mean parameter.
        :param var_noise_value: [float], It contains the variance of the noise of the model
        :param same_correlation: (boolean) If true, it uses the same correlations for the task
            kernel.

        :return: GPFittingGaussian
        """

        if mle:
            if random_seed is not None:
                np.random.seed(random_seed)

            gp = cls(type_kernel, training_data, dimensions, bounds_domain=bounds_domain,
                     thinning=thinning, n_burning=n_burning, max_steps_out=max_steps_out,
                     type_bounds=type_bounds, random_seed=random_seed, training_name=training_name,
                     problem_name=problem_name, kernel_values=kernel_values, mean_value=mean_value,
                     var_noise_value=var_noise_value, simplex_domain=simplex_domain,
                     **{SAME_CORRELATION: same_correlation})

            return gp.fit_gp_regression()

        return cls(type_kernel, training_data, dimensions, bounds_domain=bounds_domain,
                   thinning=thinning, n_burning=n_burning, max_steps_out=max_steps_out,
                   type_bounds=type_bounds, random_seed=random_seed, training_name=training_name,
                   problem_name=problem_name, kernel_values=kernel_values, mean_value=mean_value,
                   var_noise_value=var_noise_value, simplex_domain=simplex_domain,
                   **{SAME_CORRELATION: same_correlation})

    def evaluate_cross_cov(self, points_1, points_2, parameters_kernel):
        """
        Evaluate the cross covariance of the kernel of the model.

        :param points_1: np.array(nxk)
        :param points_2: np.array(mxk)
        :param parameters_kernel: np.array(l)
        :return: np.array(nxm)
        """

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            inputs_1 = separate_numpy_arrays_in_lists(points_1, self.kernel_dimensions[1])
            inputs_dict_1 = {}
            for index, input in enumerate(inputs_1):
                inputs_dict_1[self.type_kernel[index + 1]] = input

            inputs_2 = separate_numpy_arrays_in_lists(points_2, self.kernel_dimensions[1])
            inputs_dict_2 = {}
            for index, input in enumerate(inputs_2):
                inputs_dict_2[self.type_kernel[index + 1]] = input

            cov = self.class_kernel.evaluate_cross_cov_defined_by_params(
                separate_numpy_arrays_in_lists(parameters_kernel, self.number_parameters[1]),
                inputs_dict_1, inputs_dict_2,
                self.dimensions[1:], self.type_kernel[1:], **self.additional_kernel_parameters)
        elif self.type_kernel[0] == SCALED_KERNEL:
            cov = self.class_kernel.evaluate_cross_cov_defined_by_params(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                *([self.type_kernel[1]],)
            )
        else:
            cov = self.class_kernel.evaluate_cross_cov_defined_by_params(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                **self.additional_kernel_parameters
            )

        return cov

    def evaluate_grad_cross_cov_respect_point(self, points_1, points_2, parameters_kernel):
        """
        Evaluate the gradient of the cross covariance of the kernel of the model respect to
        points_1.

        :param points_1: np.array(1xk)
        :param points_2: np.array(mxk)
        :param parameters_kernel: np.array(l)
        :return: np.array(mxk)
        """

        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            grad = self.class_kernel.evaluate_grad_respect_point(
                separate_numpy_arrays_in_lists(parameters_kernel, self.number_parameters[1]),
                points_1, points_2,
                self.dimensions[1:], self.type_kernel[1:], **self.additional_kernel_parameters)
        elif self.type_kernel[0] == SCALED_KERNEL:
            grad = self.class_kernel.evaluate_grad_respect_point(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                *([self.type_kernel[1]],)
            )
        else:
            grad = self.class_kernel.evaluate_grad_respect_point(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                **self.additional_kernel_parameters
            )

        return grad

    def evaluate_hessian_cross_cov_respect_point(self, points_1, points_2, parameters_kernel):
        """
        Evaluate the hessian of the cross covariance of the kernel of the model respect to
        points_1.

        :param points_1: np.array(1xk)
        :param points_2: np.array(mxk)
        :param parameters_kernel: np.array(l)
        :return: np.array(mxkxk)
        """
        if self.type_kernel[0] == PRODUCT_KERNELS_SEPARABLE:
            hessian = self.class_kernel.evaluate_hessian_respect_point(
                separate_numpy_arrays_in_lists(parameters_kernel, self.number_parameters[1]),
                points_1, points_2,
                self.dimensions[1:], self.type_kernel[1:], **self.additional_kernel_parameters)
        elif self.type_kernel[0] == SCALED_KERNEL:
            hessian = self.class_kernel.evaluate_hessian_respect_point(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                *([self.type_kernel[1]],)
            )
        else:
            hessian = self.class_kernel.evaluate_hessian_respect_point(
                parameters_kernel, points_1, points_2, self.dimensions[0],
                **self.additional_kernel_parameters
            )

        return hessian

    def _cholesky_solve_vectors_for_posterior(self, var_noise, mean, parameters_kernel,
                                              historical_points=None, historical_evaluations=None,
                                              cache=True, clear_cache=True):
        """
        Solves the system cov(historical_points) * x = historical_evaluations - mean, and returns
        the Cholesky decomposition of cov(historical_points) too.

        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(k)
        :param historical_points: np.array(nxk)
        :param historical_evaluations: np.array(n)
        :param cache: (boolean) get cached data only if cache is True

        :return: {
            'chol': np.array(nxn),
            'solve': np.array(n)
        }
        """

        if historical_points is None:
            historical_points = self.data['points']

        if historical_evaluations is None:
            historical_evaluations = self.data['evaluations']

        chol, cov = self._chol_cov_including_noise(
            var_noise, parameters_kernel, historical_points=historical_points, cache=cache,
            clear_cache=clear_cache)


        if cache:
            cached_solve = self._get_cached_data((var_noise, tuple(parameters_kernel), mean),
                                                 SOL_CHOL_Y_UNBIASED, cache=cache)
        else:
            cached_solve = False

        if cached_solve is False:
            y_unbiased = historical_evaluations - mean
            solve = cho_solve(chol, y_unbiased)
            if cache:
                self._updated_cached_data((var_noise, tuple(parameters_kernel), mean), solve,
                                          SOL_CHOL_Y_UNBIASED, clear_cache=clear_cache)
        else:
            solve = cached_solve

        return {
            'chol': chol,
            'solve': solve,
        }

    def compute_posterior_parameters(self, points, var_noise=None, mean=None,
                                     parameters_kernel=None, only_mean=False):
        """
        Compute the posterior mean and cov of the GP at points:
            f(points) ~ GP(mu_n(points), cov_n(points, points))

        :param points: np.array(nxm)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(k)
        :param only_mean: boolean
        :return: {
            'mean': np.array(n),
            'cov': np.array(nxn)
        }
        """
        # TODO: cache solve, and np.dot(vec_cov, solve_2). We can just save it here with some
        # TODO: names like: self.mean_product = vec_cov

        if var_noise is None:
            var_noise = self.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.kernel.hypers_values_as_array

        if mean is None:
            mean = self.mean.value[0]

        chol_solve = self._cholesky_solve_vectors_for_posterior(var_noise, mean, parameters_kernel)
        chol = chol_solve['chol']
        solve = chol_solve['solve']

        vec_cov = self.evaluate_cross_cov(points, self.data['points'], parameters_kernel)

        mu_n = mean + np.dot(vec_cov, solve)

        if only_mean:
            return {
                'mean': mu_n,
                'cov': None,
            }

        if points.shape[0] == 1:
            index = (tuple(points[0, :]), tuple(parameters_kernel))

        if points.shape[0] == 1 and index in self.cache_cov_n:
            cov_n = self.cache_cov_n[index]
        else:
            solve_2 = cho_solve(chol, vec_cov.transpose())
            cov_n = self.evaluate_cov(points, parameters_kernel) - np.dot(vec_cov, solve_2)

            if points.shape[0] == 1:
                self.cache_cov_n = {}
                self.cache_cov_n[index] = cov_n

        return {
            'mean': mu_n,
            'cov': cov_n,
        }

    def gradient_posterior_parameters(self, point, var_noise=None, mean=None,
                                      parameters_kernel=None, parallel=True, only_mean=False):
        """
        Computes the gradient of the posterior parameters of the GP.

        :param point: np.array(1xn)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)

        :return: {'mean': np.array(n), 'cov': np.array(n)}
        """

        # We assume that cov(x, x) is constant respect to x (it's a radial kernel)
        # TODO: WE CAN CACHE VEC_COV

        if var_noise is None:
            var_noise = self.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.kernel.hypers_values_as_array

        if mean is None:
            mean = self.mean.value[0]

        chol_solve = self._cholesky_solve_vectors_for_posterior(var_noise, mean, parameters_kernel)
        chol = chol_solve['chol']
        solve = chol_solve['solve']

        grad_cross_cov = self.evaluate_grad_cross_cov_respect_point(point, self.data['points'],
                                                                    parameters_kernel)
        grad_mu = np.dot(grad_cross_cov.transpose(), solve)

        if only_mean:
            {'mean': grad_mu, 'cov': None}

        vec_cov = self.evaluate_cross_cov(point, self.data['points'], parameters_kernel)
        solve_2 = cho_solve(chol, grad_cross_cov)
        grad_cov = -2.0 * np.dot(vec_cov, solve_2)

        return {'mean': grad_mu, 'cov': grad_cov}

    def evaluate_gradient_sample_params(self, point, random_seed=None):
        """
        Computes the gradient of EI taking a random sample of the parameters of the model.

        :param point: np.array(n)
        :return: np.array(n)
        """

        if random_seed is not None:
            np.random.seed(random_seed)
        point = point.reshape((1, len(point)))

        params = self.sample_parameters(1)[0]

        return self.gradient_posterior_parameters(
            point, var_noise=params[0], mean=params[1], parameters_kernel=params[2:],
            only_mean=True)['mean']

    def optimize_posterior_mean(self, start=None, random_seed=None, minimize=False, n_restarts=100,
                                n_best_restarts=10, parallel=True, n_treads=0, var_noise=None,
                                mean=None, parameters_kernel=None, n_samples_parameters=0,
                                start_new_chain=False, method_opt=None, maxepoch=10,
                                candidate_solutions=None, candidate_values=None):
        """
        Optimize the posterior mean.

        :param start: np.array(n)
        :param random_seed: float
        :param minimize: boolean
        :param n_restarts: int
        :param n_best_restarts: int
        :param parallel: (boolean)
        :param n_treads: (int)
        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param n_samples_parameters: int
        :param start_new_chain: boolean
        :param method_opt: str
        :return: dictionary with the results of the optimization
        """
        if candidate_solutions is not None and len(candidate_solutions) == 0:
            candidate_solutions = None

        if random_seed is not None:
            np.random.seed(random_seed)

        if method_opt is None:
            method_opt = LBFGS_NAME

        if n_samples_parameters > 0:
            method_opt = SGD_NAME

        if start_new_chain and n_samples_parameters > 0:
            self.start_new_chain()
            self.sample_parameters(DEFAULT_N_PARAMETERS)

        bounds = self.bounds

        if n_samples_parameters == 0:
            if var_noise is None:
                var_noise = self.var_noise.value[0]

            if parameters_kernel is None:
                parameters_kernel = self.kernel.hypers_values_as_array

            if mean is None:
                mean = self.mean.value[0]

            index_cache = (var_noise, mean, tuple(parameters_kernel))
        else:
            index_cache = 'mc_mean'

        if start is None:
            start_points = DomainService.get_points_domain(n_restarts, bounds,
                                                           type_bounds=self.type_bounds)

            start = np.array(start_points)
            n_restart_ = start.shape[0]

            if n_restart_ > n_best_restarts and n_best_restarts > 0:
                point_dict = {}
                for j in xrange(start.shape[0]):
                    point_dict[j] = start[j, :]
                args = (False, None, True, 0, self, DEFAULT_N_PARAMETERS)
                ei_values = Parallel.run_function_different_arguments_parallel(
                    wrapper_posterior_mean_gp_model, point_dict, *args)
                values = [ei_values[i] for i in ei_values]
                values_index = sorted(range(len(values)), key=lambda k: values[k])
                values_index = values_index[-n_best_restarts:]
                start = []
                for j in values_index:
                    start.append(point_dict[j])
                start = np.array(start)
                n_restart_ = start.shape[0]
        else:
            n_restart_ = 1

        n_restarts = n_restart_

        objective_function = wrapper_posterior_mean_gp_model
        grad_function = wrapper_gradient_posterior_mean_gp_model

        if n_samples_parameters == 0:
            #TODO: CHECK THIS
            optimization = Optimization(
                method_opt,
                objective_function,
                bounds,
                grad_function,
                minimize=False)

            args = (False, None, parallel, 0, optimization, self, n_samples_parameters)

            opt_method = wrapper_optimize

            point_dict = {}
            for j in xrange(n_restarts):
                point_dict[j] = start[j, :]
        else:

            #TODO CHANGE wrapper_objective_voi, wrapper_grad_voi_sgd TO NO SOLVE MAX_a_{n+1} in
            #TODO: parallel for the several starting points

            args_ = (self, DEFAULT_N_PARAMETERS)

            optimization = Optimization(
                method_opt,
                objective_function,
                bounds,
                wrapper_evaluate_gradient_sample_params_gp,
                minimize=False,
                full_gradient=grad_function,
                args=args_, debug=True,
                **{'maxepoch': maxepoch}
            )

            args = (False, None, parallel, 0, optimization, n_samples_parameters, self)

            #TODO: THINK ABOUT N_THREADS. Do we want to run it in parallel?

            opt_method = wrapper_sgd

            random_seeds = np.random.randint(0, 4294967295, n_restarts)
            point_dict = {}
            for j in xrange(n_restarts):
                point_dict[j] = [start[j, :], random_seeds[j]]

        optimal_solutions = Parallel.run_function_different_arguments_parallel(
            opt_method, point_dict, *args)

        maximum_values = []
        for j in xrange(n_restarts):
            maximum_values.append(optimal_solutions.get(j)['optimal_value'])

        ind_max = np.argmax(maximum_values)
        max_ = np.max(maximum_values)

        if candidate_solutions is not None:
            n = len(candidate_values)
            candidate_solutions_2 = candidate_solutions
            values = []
            point_dict = {}

            args = (False, None, parallel, 0, self, DEFAULT_N_PARAMETERS)
            for j in range(n):
                point_dict[j] = np.array(candidate_solutions_2[j])
            values = Parallel.run_function_different_arguments_parallel(
                wrapper_posterior_mean_gp_model, point_dict, *args)

            values_candidates = []
            for j in range(n):
                values_candidates.append(values[j])
            ind_max_2 = np.argmax(values_candidates)

            if np.max(values_candidates) > max_:
                solution = point_dict[ind_max_2]
                value = np.max(values_candidates)
                optimal_solutions = {}
                optimal_solutions[ind_max] = {'solution': solution, 'optimal_value': [value]}

        logger.info("Results of the optimization of the EI: ")
        logger.info(optimal_solutions.get(ind_max))

        self.optimization_results.append(optimal_solutions.get(ind_max))

        return optimal_solutions.get(ind_max)

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
            lp += parameter.log_prior(parameters[index: index + dimension])

            index += dimension

        if not np.isinf(lp):
            lp += self.log_likelihood(parameters[0], parameters[1], parameters[2:])

        return lp

    def sample_new_observations(self, point, n_samples, random_seed=None):
        """
        Sample f(point) n_samples times.

        :param point: np.array(1xn)
        :param n_samples: int
        :param random_seed: int
        :return: np.array(n_samples)
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        posterior_parameters = self.compute_posterior_parameters(point)
        mean = posterior_parameters['mean']
        var = posterior_parameters['cov']

        samples = np.random.normal(mean[0], np.sqrt(var[0, 0]), n_samples)

        return samples

    def get_historical_best_solution(self, var_noise=None, mean=None, parameters_kernel=None,
                                     noisy_evaluations=False):
        """
        Computes the best solution so far.

        :param var_noise: float
        :param mean: float
        :param parameters_kernel: np.array(l)
        :param noisy_evaluations: boolean
        :return: float
        """
        if var_noise is None:
            var_noise = self.var_noise.value[0]

        if parameters_kernel is None:
            parameters_kernel = self.kernel.hypers_values_as_array

        if mean is None:
            mean = self.mean.value[0]

        index = (var_noise, mean, tuple(parameters_kernel))
        if index in self.best_solution:
            best = self.best_solution[index]
            return best

        if not noisy_evaluations:
            evaluations = self.data['evaluations']
        else:
            points = self.data['points']
            evaluations = self.compute_posterior_parameters(
                points, var_noise, mean, parameters_kernel, only_mean=True
            )['mean']

        best = np.max(evaluations)
        self.best_solution[index] = best

        return best

    def clean_cache(self):
        """
        Cleans the cache
        """
        self.cache_chol_cov = {}
        self.cache_sol_chol_y_unbiased = {}
        self.best_solution = {}
        self.cache_cov_n = {}

    def write_debug_data(self, problem_name, model_type, training_name, n_training, random_seed,
                         method=EI_METHOD, n_samples_parameters=0):
        """
        Write information about the different optimizations realized.

        :param problem_name: (str)
        :param model_type: (str)
        :param training_name: (str)
        :param n_training: (int)
        :param random_seed: (int)
        :param method: (str)
        :param n_samples_parameters: int
        """

        pass


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
        solve_1 = cho_solve(chol, grad_cov)

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
        solve = cho_solve(chol, -1.0 * np.ones(n))
        return -1.0 * np.dot(y_unbiased, solve)


class ValidationGPModel(object):
    _validation_filename = 'validation_kernel_{problem}_{type_kernel}_{n_training}_' \
                           '{random_seed}.json'.format
    _validation_filename_plot = 'validation_kernel_mean_vs_observations_{problem}_{type_kernel}_' \
                                '{n_training}_{random_seed}.png'.format
    _validation_filename_histogram = 'validation_kernel_histogram_{problem}_{type_kernel}_' \
                                     '{n_training}_{random_seed}.png'.format

    @classmethod
    def cross_validation_mle_parameters(cls, type_kernel, training_data, dimensions, problem_name,
                                        bounds_domain=None, thinning=0, n_burning=0,
                                        max_steps_out=1, start=None, random_seed=None,
                                        training_name=None, **kernel_parameters):
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
        :param bounds_domain: [[float, float]], lower bound and upper bound for each entry. This
            parameter is used to compute priors in a smart way.
        :param thinning: (int) Parameters of the MCMC. If it's 1, we take all the samples.
        :param n_burning: (int) Number of burnings samples for the MCMC.
        :param max_steps_out: (int) Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param start: (np.array(n)) starting point of the optimization of the llh.
        :param random_seed: int
        :param training_name: (str)
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: {
            'means': (np.array(n)), vector with the means of the GP at the points chosen in each of
                the test folds.
            'std_vec': (np.array(n)), vector with the std of the GP at the points chosen in each of
                the test folds.
            'y_eval': np.array(n), vector with the actual values of the function at the chosen
                points.
            'n_data': int, total number of points.
            'filename_plot': str,
            'filename_histogram': str,
        }
        """

        same_correlation = kernel_parameters.get(SAME_CORRELATION, False)

        kernel_name = ''
        for kernel in type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]
        kernel_name += '_same_correlation_' + str(same_correlation)

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
            else:
                training_data_sets[i]['var_noise'] = []
                test_points[i]['var_noise'] = []

        args = (False, None, True, 0, GPFittingGaussian, type_kernel, dimensions, bounds_domain,
                thinning, n_burning, max_steps_out, random_seed, problem_name, training_name)
        gp_results = Parallel.run_function_different_arguments_parallel(
            wrapper_GPFittingGaussian, training_data_sets, *args, **kernel_parameters
        )

        for i in xrange(n_data):
            if gp_results.get(i) is None:
                logger.info("It wasn't possible to create the GP instance for fold %d" % i)
                continue
            gp_objects[i] = gp_results[i]


        kwargs = {
            'start': start,
            'random_seed': random_seed,
        }

        new_gp_objects = Parallel.run_function_different_arguments_parallel(
            wrapper_fit_gp_regression, gp_objects, all_success=False, **kwargs)

        number_correct = 0
        success_runs = 0
        correct = {}

        means = np.zeros(n_data)
        std_vec = np.zeros(n_data)
        y_eval = np.zeros(n_data)

        for i in xrange(n_data):
            if new_gp_objects.get(i) is None:
                logger.info("It wasn't possible to fit the GP for %d" % i)
                continue
            success_runs += 1
            posterior = new_gp_objects[i].compute_posterior_parameters(test_points[i]['points'])

            cov = posterior['cov']
            mean = posterior['mean']

            means[i] = mean[0]
            std_vec[i] = np.sqrt(cov[0, 0])
            y_eval[i] = test_points[i]['evaluations']

            if noise:
                correct[i] = cls.check_value_within_ci(
                    test_points[i]['evaluations'], mean[0], cov[0, 0],
                    var_noise=test_points[i]['var_noise'])
            else:
                correct[i] = cls.check_value_within_ci(
                    test_points[i]['evaluations'], mean[0], cov[0, 0])
            if correct[i]:
                number_correct += 1

        if success_runs != 0:
            proportion_success = number_correct / float(success_runs)
            logger.info("Proportion of success is %f" % proportion_success)
        else:
            proportion_success = -1
            logger.info("All runs failed!!")

        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)

        if not os.path.exists(DIAGNOSTIC_KERNEL_DIR):
            os.mkdir(DIAGNOSTIC_KERNEL_DIR)

        diag_dir = path.join(DIAGNOSTIC_KERNEL_DIR, problem_name)

        if not os.path.exists(diag_dir):
            os.mkdir(diag_dir)

        filename = path.join(diag_dir, cls._validation_filename(
            problem=problem_name,
            type_kernel=kernel_name,
            n_training=n_data,
            random_seed=random_seed,
        ))


        filename_plot = path.join(diag_dir, cls._validation_filename_plot(
            problem=problem_name,
            type_kernel=kernel_name,
            n_training=n_data,
            random_seed=random_seed,
        ))

        filename_histogram = path.join(diag_dir, cls._validation_filename_histogram(
            problem=problem_name,
            type_kernel=kernel_name,
            n_training=n_data,
            random_seed=random_seed,
        ))

        results = {
            'means': means,
            'std_vec': std_vec,
            'y_eval': y_eval,
            'n_data': n_data,
            'success_proportion': proportion_success,
            'filename_plot': filename_plot,
            'filename_histogram': filename_histogram,
            'number_correctly_fitted_models': success_runs,
        }

        JSONFile.write(results, filename)

        return results

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
