from __future__ import absolute_import

import numpy as np

from scipy.spatial.distance import cdist

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
    convert_dictionary_gradient_to_simple_dictionary,
)
from stratified_bayesian_optimization.lib.constant import (
    ORNSTEIN_KERNEL,
    LENGTH_SCALE_ORNSTEIN_NAME,
    SIGMA2_NAME,
    LARGEST_NUMBER,
    SMALLEST_POSITIVE_NUMBER,
)
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.priors.log_normal_square import LogNormalSquare


class Swk(AbstractKernel):

    def __init__(self, alpha, beta, **kernel_parameters):
        """
        :param dimension: int
        :param sigma: ParameterEntity
        """

        name = 'swk'

        super(Swk, self).__init__(name, 1, 2)

        self.alpha = alpha
        self.beta = beta

    @property
    def hypers(self):
        return {
            self.alpha.name: self.alpha,
            self.beta.name: self.beta,
        }

    @property
    def hypers_as_list(self):
        """
        This function defines the default order of the parameters.
        :return: [ParameterEntity]
        """
        return [self.alpha, self.beta]

    @property
    def hypers_values_as_array(self):
        """

        :return: np.array(n)
        """
        parameters = []
        parameters.append(self.alpha.value)
        parameters.append(self.beta.value)

        return np.concatenate(parameters)

    def sample_parameters(self, number_samples, random_seed=None):
        """

        :param number_samples: (int) number of samples
        :param random_seed: int
        :return: np.array(number_samples x k)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        samples = []
        parameters = [self.alpha, self.beta]
        for parameter in parameters:
            samples.append(parameter.sample_from_prior(number_samples))
        return np.concatenate(samples, 1)

    def get_bounds_parameters(self):
        """
        Return bounds of the parameters of the kernel
        :return: [(float, float)]
        """
        bounds = []
        parameters = [self.alpha, self.beta]
        for parameter in parameters:
            bounds += parameter.bounds
        return bounds

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return [(self.alpha.name, None), (self.beta.name, None)]

    def set_parameters(self, alpha=None, beta=None):
        """

        :param length_scale: ParameterEntity
        """
        if alpha is not None:
            self.alpha = alpha

        if beta is not None:
            self.beta = beta

    def update_value_parameters(self, params):
        """

        :param params: np.array(n)
        """
        self.alpha.set_value(params[0])
        self.beta.set_value(params[1])

    @classmethod
    def define_kernel_from_array(cls, dimension, params, **kernel_parameters):
        """
        :param dimension: (int) dimension of the domain of the kernel
        :param params: (np.array(k)) The first part are the parameters for length_scale.

        :return: Matern52
        """

        alpha = ParameterEntity('alpha', [params[0]], None)
        beta = ParameterEntity('beta', [params[1]], None)


        return cls(alpha, beta)

    @classmethod
    def define_default_kernel(cls, dimension, bounds=None, default_values=None,
                              parameters_priors=None, **kernel_parameters):
        """
        :param dimension: (int) dimension of the domain of the kernel
        :param bounds: [[float, float]], lower bound and upper bound for each entry of the domain.
            This parameter is used to compute priors in a smart way.
        :param default_values: (np.array(k)) The first part are the parameters for length_scale
        :param parameters_priors: {
            LENGTH_SCALE_NAME: [float],
        }

        :return: Matern52
        """

        if parameters_priors is None:
            parameters_priors = {}

        if default_values is None:
            alpha = parameters_priors.get('alpha', [1.0])
            default_values = alpha

            default_value_sigma = parameters_priors.get('beta', [1.0])
            default_values += default_value_sigma

        kernel = cls.define_kernel_from_array(dimension, default_values)


        if bounds is None:
            bounds = [[0.1, 10.0], [0.1, 100.0]]

        kernel.alpha.bounds = bounds[0]
        kernel.beta.bounds = bounds[1]

        prior = UniformPrior(1, 1 * [0.1], [10.0])
        prior_2 = UniformPrior(1, 1 * [0.1], [100.0])

        kernel.alpha.prior = prior
        kernel.beta.prior = prior_2



        return kernel

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
        inputs_1 = np.array(inputs_1)
        inputs_2 = np.array(inputs_2)

        n = inputs_1.shape[0]
        n2 = inputs_2.shape[0]
        covM = np.zeros((n, n2))
        #
        # print n, n2
        # print inputs_1
        for i in range(n):
            for j in range(n2):
                a1 = float(inputs_1[i, 0])
                a2 = float(inputs_2[j, 0])

                value = (self.beta.value[0] / (a1 + a2 + self.beta.value[0]))
                covM[i, j] = value ** (self.alpha.value[0])

        return covM

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: {
            'length_scale': {'entry (int)': nxn},
        }
        """
        pass

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        pass


    def hessian_respect_point(self, point, inputs):
        """
        Computes the hessians of cov(point, inputs) respect point

        :param point:
        :param inputs:
        :return: np.array(nxdxd)
        """
        pass

    @classmethod
    def evaluate_grad_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the gradient of the kernel defined by params respect to the point.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)
        :param dimension: (int) dimension of the domain of the kernel
        :return: np.array(nxd)

        """
        pass

    @classmethod
    def evaluate_hessian_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the hessian of the kernel defined by params respect to the point.

        :param params:
        :param point:
        :param inputs:
        :param dimension: int
        :return:
        """
        pass

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: (np.array(nxn)) cov(inputs) where the kernel is defined with params
        """
        kernel = cls.define_kernel_from_array(dimension, params)

        return kernel.cov(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """
        pass

    @classmethod
    def evaluate_cross_cov_defined_by_params(cls, params, inputs_1, inputs_2, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale.
        :param inputs_1: np.array(nxm)
        :param inputs_2: np.array(kxm)
        :param dimension: (int) dimension of the domain of the kernel

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """
        kernel = cls.define_kernel_from_array(dimension, params)
        return kernel.cross_cov(inputs_1, inputs_2)

    @staticmethod
    def define_prior_parameters(data, dimension):
        """
        Defines value of the parameters of the prior distributions of the kernel's parameters.

        :param data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}. Each point is the is an index of the task.
        :param dimension: int
        :return:  {
            LENGTH_SCALE_NAME: [float],
        }
        """
        # Take mean value of |x-y| for all points x,y in the training data. I think that it's a
        # good starting value for the parameter ls in the kernel.

        return {
            'alpha': [1.0],
            'beta': [1.0],
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
        pass

    @staticmethod
    def parameters_from_list_to_dict(params, **kwargs):
        """
        Converts a list of parameters to dictionary using the order of the kernel.

        :param params: [float]

        :return: {
            LENGTH_SCALE_NAME: [float],
        }
        """

        parameters = {}

        parameters['alpha'] = [params[0]]
        parameters['beta'] = [params[1]]

        return parameters


