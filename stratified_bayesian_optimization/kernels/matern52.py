from __future__ import absolute_import

import numpy as np

from collections import defaultdict

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.util import \
    convert_dictionary_gradient_to_simple_dictionary


class Matern52(AbstractKernel):

    def __init__(self, dimension, length_scale, sigma2):
        """

        :param length_scale: ParameterEntity
        :param sigma: ParameterEntity
        """

        name = 'Matern52'

        super(Matern52, self).__init__(name, dimension)

        self.length_scale = length_scale
        self.sigma2 = sigma2

    @property
    def hypers(self):
        return {
            self.length_scale.name: self.length_scale,
            self.sigma2.name: self.sigma2,
        }

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return [(self.length_scale.name, [(i, None) for i in xrange(self.dimension)]),
                (self.sigma2.name, None)]

    def set_parameters(self, length_scale=None, sigma2=None):
        """

        :param length_scale: ParameterEntity
        :param sigma: ParameterEntity
        """
        if length_scale is not None:
            self.length_scale = length_scale

        if sigma2 is not None:
            self.sigma2 = sigma2


    @classmethod
    def define_kernel_from_array(cls, dimension, params):
        """
        :param dimension: (int) dimension of the domain of the kernel
        :param params: (np.array(k)) The first part are the parameters for length_scale, the
            second part is the parameter for sigma2.

        :return: Matern52
        """

        length_scale = ParameterEntity('length_scale', params[0:dimension], None)
        sigma2 = ParameterEntity('sigma2', params[dimension:dimension+1], None)

        return cls(dimension, length_scale, sigma2)

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
        r2 = np.abs(Distances.dist_square_length_scale(self.length_scale.value, inputs_1, inputs_2))
        r = np.sqrt(r2)
        cov = (1.0 + np.sqrt(5)*r + (5.0/3.0)*r2) * np.exp(-np.sqrt(5)*r)
        return cov * self.sigma2.value

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: {
            'length_scale': {'entry (int)': nxn},
            'sigma_square': nxn,
        }
        """
        grad = GradientLSMatern52.gradient_respect_parameters_ls(inputs, self.length_scale,
                                                                 self.sigma2)

        grad[self.sigma2.name] = self.cov(inputs) / self.sigma2.value

        return grad

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        grad = GradientLSMatern52.grad_respect_point(self.length_scale, self.sigma2, point, inputs)

        return grad

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale, the
            second part is the parameter for sigma2.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: cov(inputs) where the kernel is defined with params
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        return matern52.cov(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k)) The first part are the parameters for length_scale, the
            second part is the parameter for sigma2.
        :param inputs: np.array(nxm)
        :param dimension: (int) dimension of the domain of the kernel
        :return: {
            'length_scale': {'entry (int)': nxn},
            'sigma_square': nxn,
        }
        """
        matern52 = cls.define_kernel_from_array(dimension, params)
        gradient = matern52.gradient_respect_parameters(inputs)

        names = matern52.name_parameters_as_list

        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)
        return gradient


class GradientLSMatern52(object):

    @classmethod
    def gradient_respect_parameters_ls(cls, inputs, ls, sigma2):
        """

        :param inputs: np.array(nxd)
        :param ls: (ParameterEntity) length_scale
        :param sigma2: (ParameterEntity)
        :return: {
            'length_scale': {'entry (int)': nxn}
        }
        """

        derivate_respect_to_r = cls.gradient_respect_distance(ls, sigma2, inputs)

        grad = {}
        grad[ls.name] = {}

        grad_distance_ls = Distances.gradient_distance_length_scale_respect_ls(ls.value, inputs)

        for i in range(ls.dimension):
            grad[ls.name][i] = grad_distance_ls[i] * derivate_respect_to_r

        return grad

    @classmethod
    def gradient_respect_distance(cls, ls, sigma2, inputs):
        """
        :param ls: (ParameterEntity) length_scale
        :param sigma2: (ParameterEntity)
        :param inputs: np.array(nxd)
        :return: np.array(nxn)
        """

        return cls.gradient_respect_distance_cross(ls, sigma2, inputs, inputs)

    @classmethod
    def gradient_respect_distance_cross(cls, ls, sigma2, inputs_1, inputs_2):
        """

        :param ls: (ParameterEntity) length_scale
        :param sigma2: (ParameterEntity)
        :param inputs_1: np.array(nxd)
        :param inputs_2: np.array(mxd)
        :return: np.array(nxm)
        """
        r2 = np.abs(Distances.dist_square_length_scale(ls.value, inputs_1, inputs_2))
        r = np.sqrt(r2)

        part_1 = ((1.0 + np.sqrt(5)*r + (5.0/3.0)*r2) * np.exp (-np.sqrt(5)* r) * (-np.sqrt(5)))
        part_2 = (np.exp (-np.sqrt(5)* r) * (np.sqrt(5) + (10.0/3.0) * r))
        derivate_respect_to_r = part_1 + part_2
        return derivate_respect_to_r * sigma2.value

    @classmethod
    def grad_respect_point(cls, ls, sigma2, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param ls: (ParameterEntity) length_scale
        :param sigma: (ParameterEntity)
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        ##TO DO: GENERALIZE FOR THE CASE WHEN W isn't discrete and does depend on x?

        derivate_respect_to_r = cls.gradient_respect_distance_cross(ls, sigma2, point, inputs)
        grad_distance_point = \
            Distances.gradient_distance_length_scale_respect_point(ls.value, point, inputs)

        gradient = grad_distance_point * derivate_respect_to_r.transpose()

        return gradient
