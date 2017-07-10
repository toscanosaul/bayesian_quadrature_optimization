from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_gradient_to_simple_dictionary,
)
from stratified_bayesian_optimization.lib.constant import (
    SIGMA2_NAME,
    SMALLEST_POSITIVE_NUMBER,
)
from stratified_bayesian_optimization.lib.util_kernels import (
    find_kernel_constructor,
)
from stratified_bayesian_optimization.priors.log_normal_square import LogNormalSquare


class ScaledKernel(AbstractKernel):

    def __init__(self, dimension, kernel, sigma2):
        """
        :param dimension: (int) dimension of the kernel parameter
        :param kernel: kernel instance
        :param sigma2: ParameterEntity
        """

        name = kernel.name
        dimension_parameters = 1 + kernel.dimension_parameters

        super(ScaledKernel, self).__init__(name, dimension, dimension_parameters)

        self.sigma2 = sigma2
        self.kernel = kernel

    @property
    def hypers(self):
        parameters = self.kernel.hypers
        parameters[self.sigma2.name] = self.sigma2
        return parameters

    @property
    def hypers_as_list(self):
        """
        This function defines the default order of the parameters.
        :return: [ParameterEntity]
        """
        return self.kernel.hypers_as_list + [self.sigma2]

    @property
    def hypers_values_as_array(self):
        """
        :return: np.array(n)
        """

        parameters = list(self.kernel.hypers_values_as_array)
        parameters.append(self.sigma2.value[0])

        return np.array(parameters)

    def sample_parameters(self, number_samples, random_seed=None):
        """

        :param number_samples: (int) number of samples
        :param random_seed: int
        :return: np.array(number_samples x k)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        samples = []
        parameters = self.hypers_as_list
        for parameter in parameters:
            samples.append(parameter.sample_from_prior(number_samples))
        return np.concatenate(samples, 1)

    def get_bounds_parameters(self):
        """
        Return bounds of the parameters of the kernel
        :return: [(float, float)]
        """
        bounds = []
        parameters = self.hypers_as_list
        for parameter in parameters:
            bounds += parameter.bounds
        return bounds

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return self.kernel.name_parameters_as_list + [(self.sigma2.name, None)]

    def set_parameters(self, kernel_parameters=None, sigma2=None):
        """
        :param kernel_parameters: [ParameterEntity]
        :param sigma2: ParameterEntity
        """

        if kernel_parameters is not None:
            self.kernel.set_parameters(*kernel_parameters)

        if sigma2 is not None:
            self.sigma2 = sigma2

    def update_value_parameters(self, params):
        """

        :param params: np.array(n)
        """
        self.kernel.update_value_parameters(params[0: -1])
        self.sigma2.set_value(params[-1:])

    @classmethod
    def define_kernel_from_array(cls, dimension, params, *args):
        """
        :param dimension: (int) dimension of the kernel instance used in this kernel
        :param params: (np.array(k)) The first part are the parameters for the kernel, the
            second part is the parameter for sigma2.
        :param args: [str] name of the kernel instance

        :return: scaled kernel
        """

        for name in args[0]:
            kernel_ct = find_kernel_constructor(name)
            kernel = kernel_ct.define_kernel_from_array(dimension, params[0: -1])

        sigma2 = ParameterEntity(SIGMA2_NAME, params[-1:], None)

        return cls(dimension, kernel, sigma2)

    @classmethod
    def define_default_kernel(cls, dimension, bounds=None, default_values=None,
                              parameters_priors=None, *args):
        """
        :param dimension: (int) dimension of the domain of the kernel instance
        :param bounds: [[float, float]], lower bound and upper bound for each entry of the domain.
            This parameter is used to compute priors in a smart way.
        :param default_values: (np.array(k)) The first part are the parameters for the instance of
            the kernel, the second part is the parameter for sigma2.
        :param parameters_priors: {
            PARAMATER_NAME: [float],
            SIGMA2_NAME: float,
        }
        :param args: [str] List with the names of the default kernel

        :return: scaled kernel
        """

        if parameters_priors is None:
            parameters_priors = {}

        default_values_kernel = None

        if default_values is not None:
            default_values_kernel = default_values[0: -1]
            default_value_sigma = default_values[-1:]
        else:
            default_value_sigma = [parameters_priors.get(SIGMA2_NAME, 1.0)]

        for name in args[0]:
            kernel_ct = find_kernel_constructor(name)
            kernel = kernel_ct.define_default_kernel(dimension, bounds, default_values_kernel,
                                                     parameters_priors)

        sigma2 = ParameterEntity(SIGMA2_NAME, default_value_sigma,
                                 LogNormalSquare(1, 1.0, np.sqrt(default_value_sigma)),
                                 bounds=[(SMALLEST_POSITIVE_NUMBER, None)])

        kernel_scaled = cls(dimension, kernel, sigma2)

        return kernel_scaled

    def cov(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: np.array(nxn)
        """
        return self.cross_cov(inputs, inputs)

    def cross_cov(self, inputs_1, inputs_2):
        """

        :param inputs_1: np.array(nxd)
        :param inputs_2: np.array(mxd)
        :return: np.array(nxm)
        """

        return self.kernel.cross_cov(inputs_1, inputs_2) * self.sigma2.value

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: {
            Parameter_Name: value,
            'sigma_square': nxn,
        }
        """
        grad = self.kernel.gradient_respect_parameters(inputs)
        for element in grad:
            if type(grad[element]) == dict:
                for i in grad[element]:
                    grad[element][i] *= self.sigma2.value
            else:
                grad[element] *= self.sigma2.value
        grad[self.sigma2.name] = self.cov(inputs) / self.sigma2.value

        return grad

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        grad = self.kernel.grad_respect_point(point, inputs)

        return grad * self.sigma2.value

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension,  *args):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for the kernel instance, the
            second part is the parameter for sigma2.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :param args: [str] List with the names of the kernels.

        :return: cov(inputs) where the kernel is defined with params
        """
        kernel = cls.define_kernel_from_array(dimension, params, *args)

        return kernel.cov(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension, *args):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for the kernel instance, the
            second part is the parameter for sigma2.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :param args: [str] List with the names of the kernels.

        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """
        kernel = cls.define_kernel_from_array(dimension, params, *args)

        gradient = kernel.gradient_respect_parameters(inputs)

        names = kernel.name_parameters_as_list

        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)
        return gradient

    @classmethod
    def evaluate_cross_cov_defined_by_params(cls, params, inputs_1, inputs_2, dimension, *args):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for the kernel instance, the
            second part is the parameter for sigma2.
        :param inputs_1: np.array(nxm)
        :param inputs_2: np.array(kxm)
        :param dimension: (int) dimension of the domain of the kernel
        :param args: [str] List with the names of the kernels.

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args)
        return kernel.cross_cov(inputs_1, inputs_2)

    @staticmethod
    def define_prior_parameters(data, dimension, var_evaluations=None):
        """
        Defines value of the parameters of the prior distributions of the kernel's parameters.

        :param data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}. Each point is the is an index of the task.
        :param dimension: int
        :param var_evaluations: (float), an estimator for the sigma2 parameter.
        :return:  {
            SIGMA2_NAME: float,
        }
        """

        if var_evaluations is None:
            var_evaluations = np.var(data['evaluations'])

        return {
            SIGMA2_NAME: var_evaluations,
        }

    @staticmethod
    def compare_kernels(kernel1, kernel2):
        """
        Compare the values of kernel1 and kernel2. Returns True if they're equal, otherwise it
        return False.

        :param kernel1: Matern52 instance object
        :param kernel2: Matern52 instance object
        :return: boolean
        """
        if kernel1.name != kernel2.name:
            return False

        if kernel1.dimension != kernel2.dimension:
            return False

        if kernel1.dimension_parameters != kernel2.dimension_parameters:
            return False

        if kernel1.sigma2.value != kernel2.sigma2.value:
            return False

        kernel_ = kernel1.kernel
        kernel_2 = kernel2.kernel

        ct = find_kernel_constructor(kernel_.name)

        return ct.compare_kernels(kernel_, kernel_2)

    @staticmethod
    def parameters_from_list_to_dict(params, **kwargs):
        """
        Converts a list of parameters to dictionary using the order of the kernel.

        :param params: [float]
        :param kwargs:{
            'dimensions': [float],
            'kernel': str,
        }

        :return: {
            KERNEL_PARAMETERS: [float],
            SIGMA2_NAME: float,
        }
        """

        ct = find_kernel_constructor(kwargs['kernels'])
        parameters = ct.parameters_from_list_to_dict(params[0: -1])
        parameters[SIGMA2_NAME] = params[-1]

        return parameters
