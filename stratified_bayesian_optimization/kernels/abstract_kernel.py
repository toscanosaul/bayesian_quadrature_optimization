from __future__ import absolute_import

from abc import ABCMeta, abstractmethod


class AbstractKernel(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, dimension, dimension_parameters):
        """

        :param name: str
        :param dimension: int
        :param dimension_parameters: int
        """

        self.name = name
        self.dimension = dimension
        self.dimension_parameters = dimension_parameters

    @property
    def hypers(self):
        raise NotImplementedError("Not implemented")

    @property
    def hypers_as_list(self):
        raise NotImplementedError("Not implemented")

    @property
    def hypers_values_as_array(self):
        raise NotImplementedError("Not implemented")

    @property
    def name_parameters_as_list(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def set_parameters(self, *params):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_value_parameters(self, params):
        raise NotImplementedError("Not implemented")

    @classmethod
    @abstractmethod
    def define_kernel_from_array(cls, dimension, params):
        raise NotImplementedError("Not implemented")

    @classmethod
    @abstractmethod
    def define_default_kernel(cls, dimension, bounds, default_values):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def cov(self, inputs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def cross_cov(self, inputs_1, inputs_2):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def gradient_respect_parameters(self, inputs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def grad_respect_point(self, point, inputs):
        raise NotImplementedError("Not implemented")

    # The following two functions are useful to estimate the MLE
    @classmethod
    @abstractmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension):
        raise NotImplementedError("Not implemented")

    @classmethod
    @abstractmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def sample_parameters(self, number_samples, random_seed):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_bounds_parameters(self):
        raise NotImplementedError("Not implemented")

    @staticmethod
    @abstractmethod
    def compare_kernels(kernel1, kernel2):
        raise NotImplementedError("Not implemented")
