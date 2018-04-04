from __future__ import absolute_import

from functools import reduce

import numpy as np

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_gradient_to_simple_dictionary,
    convert_dictionary_from_names_kernels_to_only_parameters,
)
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    ORNSTEIN_KERNEL,
)
from stratified_bayesian_optimization.lib.util_kernels import (
    find_define_kernel_from_array,
    find_kernel_constructor,
)
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.ornstein import Ornstein


class ProductKernels(AbstractKernel):
    # TODO - Generaliza to more than two kernels, and cover the case where kernels are defined in
    # the same domain.

    # Possible kernels for the product
    _possible_kernels_ = [MATERN52_NAME, TASKS_KERNEL_NAME]

    def __init__(self, *kernels):
        """

        :param *kernels: ([AbstractKernel])
        """

        name = PRODUCT_KERNELS_SEPARABLE + ':'
        dimension = 0
        dimension_parameters = 0
        for kernel in kernels:
            name += kernel.name + '_'
            dimension += kernel.dimension
            dimension_parameters += kernel.dimension_parameters

        super(ProductKernels, self).__init__(name, dimension, dimension_parameters)

        self.kernels = {}
        self.parameters = {}

        self.names = [kernel.name for kernel in kernels]

        for kernel in kernels:
            self.kernels[kernel.name] = kernel
            self.parameters[kernel.name] = self.kernels[kernel.name].hypers

    @property
    def hypers(self):
        return self.parameters

    @property
    def hypers_as_list(self):
        """
        This function defines the default order of the parameters.
        :return: [ParameterEntity]
        """
        parameters = []
        for name in self.names:
            parameters += self.kernels[name].hypers_as_list

        return parameters

    @property
    def hypers_values_as_array(self):
        """

        :return: np.array(n)
        """
        parameters = []
        for name in self.names:
            parameters.append(self.kernels[name].hypers_values_as_array)
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
        for name in self.names:
            samples.append(self.kernels[name].sample_parameters(number_samples))
        return np.concatenate(samples, 1)

    def get_bounds_parameters(self):
        """
        Return bounds of the parameters of the kernel
        :return: [(float, float)]
        """
        bounds = []
        for name in self.names:
            bounds += self.kernels[name].get_bounds_parameters()
        return bounds

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        names = []
        for name in self.names:
            tmp_names = self.kernels[name].name_parameters_as_list
            tmp_names = [name_tmp for name_tmp in tmp_names]
            names += tmp_names
        return names

    @classmethod
    def define_kernel_from_array(cls, dimension, params, *args, **kernel_parameters):
        """
        :param dimension: [int] list with the dimensions of the kernel
        :param params: [np.array(k)] The first part are related to the parameters of the first
            kernel and so on.
        :param args: ([str], ) List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: ProductKernels
        """

        kernels = []

        for name, dim, param in zip(args[0], dimension, params):
            kernel_ct = find_define_kernel_from_array(name)
            kernels.append(kernel_ct(dim, param, **kernel_parameters))

        return cls(*kernels)

    @classmethod
    def define_default_kernel(cls, dimension, bounds=None, default_values=None,
                              parameters_priors=None, *args, **kernel_parameters):
        """
        :param dimension: [(int)] dimension of the domain of the kernels. It's the number of tasks
            for the tasks kernel.
        :param default_values: [np.array(n)] List with the default value for the parameters of each
            of the kernels of the product.
        :param bounds: [[[float], float]] List witht he bounds of the domain of each of the kernels
            of the product.
        :param parameters_priors: {
                SIGMA2_NAME: float
                LENGTH_SCALE_NAME: [float]
                LOWER_TRIANG_NAME: [float]
            }
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: ProductKernels
        """
        kernels = []

        if default_values is None:
            iterate = zip(args[0], dimension)
        else:
            iterate = zip(args[0], dimension, default_values)

        if bounds is None:
            bounds = len(dimension) * [None]

        index = 0
        for value in iterate:
            name = value[0]
            dim = value[1]
            bound = bounds[index]
            index += 1
            constructor = find_kernel_constructor(name)
            if default_values is None:
                kernels.append(constructor.define_default_kernel(dim, bound, None,
                                                                 parameters_priors,
                                                                 **kernel_parameters))
            else:
                kernels.append(constructor.define_default_kernel(dim, bound, value[2],
                                                                 parameters_priors,
                                                                 **kernel_parameters))

        return cls(*kernels)

    def set_parameters(self, parameters):
        """

        :param parameters: {(str) kernel_name: {(str) parameter_name (as in set_parameter function
            of kernel) : ParameterEntity}}
        """

        for name in self.names:
            if name in parameters:
                self.kernels[name].set_parameters(**parameters[name])
                self.parameters[name] = self.kernels[name].hypers

    def update_value_parameters(self, params):
        """
        :param params: np.array(n)
        """
        index = 0
        for name in self.names:
            dim_param = self.kernels[name].dimension_parameters
            self.kernels[name].update_value_parameters(
                params[index: index + dim_param])
            index += dim_param

    def cov(self, inputs):
        """

        :param inputs: np.array(nxd)
        :return: np.array(nxn)
        """

        return self.cross_cov(inputs, inputs)

    def cross_cov(self, inputs_1, inputs_2):
        """

        :param inputs_1: np.array(nxd)
        :param inputs_2:  np.array(mxd)
        :return: np.array(nxm)
        """

        inputs_1_dict = self.inputs_from_array_to_dict(inputs_1)
        inputs_2_dict = self.inputs_from_array_to_dict(inputs_2)

        return self.cross_cov_dict(inputs_1_dict, inputs_2_dict)

    def inputs_from_array_to_dict(self, inputs):
        inputs_1_dict = {}
        cont = 0
        for name in self.names:
            inputs_1_dict[name] = inputs[:, cont: cont + self.kernels[name].dimension]
            cont += self.kernels[name].dimension

        return inputs_1_dict

    def cov_dict(self, inputs):
        """

        :param inputs: {(str) kernel_name: np.array(nxd)}
        :return: np.array(nxn)
        """

        return self.cross_cov_dict(inputs, inputs)

    def cross_cov_dict(self, inputs_1, inputs_2):
        """

        :param inputs_1: {(str) kernel_name: np.array(nxd)}
        :param inputs_2: {(str) kernel_name: np.array(mxd')}
        :return: np.array(nxm)
        """

        return reduce(lambda K1, K2: K1 * K2,
                      [self.kernels[name].cross_cov(inputs_1[name], inputs_2[name])
                       for name in self.names])

    def gradient_respect_parameters(self, inputs):
        """
        :param inputs: {(str) kernel_name: np.array(nxd)}
        :return: {
            (str) kernel_name: { (str) parameter_name: np.array(nxn) or {'entry (int)': nxn}}
        }
        """
        # TODO - Generalize to more than two kernels

        grad = {}
        cov = {}

        for name in self.names:
            grad[name] = self.kernels[name].gradient_respect_parameters(inputs[name])
            cov[name] = self.kernels[name].cov(inputs[name])

        grad_product = {}

        for i in range(2):
            grad_product[self.names[i]] = {}
            for name_param in self.kernels[self.names[i]].hypers:
                if type(grad[self.names[i]][name_param]) == dict:
                    grad_product[self.names[i]][name_param] = {}
                    for entry in grad[self.names[i]][name_param]:
                        grad_product[self.names[i]][name_param][entry] = \
                            cov[self.names[(i + 1) % 2]] * grad[self.names[i]][name_param][entry]
                else:
                    grad_product[self.names[i]][name_param] = \
                        cov[self.names[(i + 1) % 2]] * grad[self.names[i]][name_param]

        return grad_product

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1xd)
        :param inputs: np.array(nxd)

        :return: np.array(nxd)
        """
        # TODO - Generalize when the gradients share the same inputs
        # TODO - Generalize to more than two kernels

        point_dict = self.inputs_from_array_to_dict(point)
        inputs_dict = self.inputs_from_array_to_dict(inputs)

        grad_dict = self.grad_respect_point_dict(point_dict, inputs_dict)

        grad = []
        for name in self.names:
            grad.append(grad_dict[name])

        return np.concatenate(grad, 1)

    def hessian_respect_point(self, point, inputs):
        """
        Computes the hessians of cov(point, inputs) respect point

        :param point:
        :param inputs: np.array(nxd)
        :return: np.array(nxdxd)
        """

        point_dict = self.inputs_from_array_to_dict(point)
        inputs_dict = self.inputs_from_array_to_dict(inputs)

        hessian_dict = self.hessian_respect_point_dict(point_dict, inputs_dict)

        hess = hessian_dict['hessian']
        hess_diag = hessian_dict['diagonal_hessian']

        # Only works for the product of two kernels
        name_1 = self.names[0]
        name_2 = self.names[1]

        hessian = np.concatenate([hess[name_1], hess_diag], 2)

        # I'm not 100% sure that this is correct

        hessian_2 = np.concatenate([np.einsum('ikj', hess_diag), hess[name_2]], 2)

        hessian = np.concatenate([hessian, hessian_2], axis=1)

        # for i in xrange(inputs.shape[0]):
        #     hessian[i] = [hess[name_1][i], hess_diag[i]]
        #     hessian[i] = np.concatenate(hessian[i], 1)
        #
        #     hess_2 = [hess_diag[i].transpose(), hess[name_2][i]]
        #     hess_2 = np.concatenate(hess_2, 1)
        #
        #     hessian[i] = np.concatenate((hessian[i], hess_2), axis=0)

        return hessian

    def hessian_respect_point_dict(self, point, inputs):
        """
        Computes the hessians of cov(point, inputs) respect point.

        :param point: {(str) kernel_name: np.array(1xd)}
        :param inputs: {(str) kernel_name: np.array(nxd)}

        :return: {'hessian': (str) kernel_name: np.array(nxdxd),
            'diagonal_hessian': np.array(nxdxd)}
        """
        # TODO - Generalize when the gradients share the same inputs
        # TODO - Generalize to more than two kernels

        hess = {}
        grad = {}
        cov = {}

        for name in self.names:
            hess[name] = self.kernels[name].hessian_respect_point(point[name], inputs[name])
            grad[name] = self.kernels[name].grad_respect_point(point[name], inputs[name])
            cov[name] = self.kernels[name].cross_cov(point[name], inputs[name]).transpose()

        hessian = {}
        diagonal_hessian = {}
        for j in xrange(2):
            ind_1 = j
            ind_2 = (j + 1) % 2

            name_1 = self.names[ind_1]
            name_2 = self.names[ind_2]

            part_1 = hess[name_1] * cov[name_2][:, 0][:, np.newaxis, np.newaxis]
            hessian[name_1] = part_1

            if 0 == j:
                diagonal_hessian = grad[name_1][:, :, None] *grad[name_2][:, None, :]

            # for i in xrange(total_points):
            #     part_1 = hess[name_1][i] * cov[name_2][i, 0]
            #     hessian[name_1][i] = part_1
            #
            #     if 0 == j:
            #         diagonal_part = np.dot(grad[name_1][i:i + 1, :].transpose(),
            #                                grad[name_2][i:i + 1, :])
            #         diagonal_hessian[i] = diagonal_part

        return {'hessian': hessian, 'diagonal_hessian': diagonal_hessian}

    def grad_respect_point_dict(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: {(str) kernel_name: np.array(1xd)}
        :param inputs: {(str) kernel_name: np.array(nxd)}

        :return: {(str) kernel_name: np.array(nxd)}
        """
        # TODO - Generalize when the gradients share the same inputs
        # TODO - Generalize to more than two kernels

        grad = {}
        cov = {}

        for name in self.names:
            grad[name] = self.kernels[name].grad_respect_point(point[name], inputs[name])
            cov[name] = self.kernels[name].cross_cov(point[name], inputs[name]).transpose()

        gradient = {}

        for i in range(2):
            gradient[self.names[i]] = grad[self.names[i]] * cov[self.names[(i + 1) % 2]]

        return gradient

    @classmethod
    def evaluate_grad_respect_point(cls, params, point, inputs, dimension, *args,
                                    **kernel_parameters):
        """
        Evaluate the gradient of the kernel defined by params respect to the point.

        :param params: [np.array(k)] The first part are related to the parameters of the first
            kernel and so on.
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)
        :param dimension: [int] list with the dimensions of the kernel
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
        :return: np.array(nxd)

        """
        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        return kernel.grad_respect_point(point, inputs)

    @classmethod
    def evaluate_hessian_respect_point(cls, params, point, inputs, dimension, *args,
                                    **kernel_parameters):
        """
        Evaluate the hessian of the kernel defined by params respect to the point.

        :param params: [np.array(k)] The first part are related to the parameters of the first
            kernel and so on.
        :param point: np.array(1xd)
        :param inputs: np.array(nxd)
        :param dimension: [int] list with the dimensions of the kernel
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
        :return:
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        return kernel.hessian_respect_point(point, inputs)

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension, *args, **kernel_parameters):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: [np.array(k)] The first part are related to the parameters of the first
            kernel and so on.
        :param inputs: {(str) kernel_name: np.array(nxd)}.
        :param dimension: [int] list with the dimensions of the kernel
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: cov(inputs) where the kernel is defined with params
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        return kernel.cov_dict(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension, *args,
                                                       **kernel_parameters):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param dimension: [int] list with the dimensions of the kernel
        :param params: [np.array(k)] The first part are related to the parameters of the first
            kernel and so on.
        :param inputs: {(str) kernel_name: np.array(nxd)}
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
        :return: {
            (int) i: np.array(nxn), derivative respect to the ith parameter
        }
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        gradient = kernel.gradient_respect_parameters(inputs)

        gradient = convert_dictionary_from_names_kernels_to_only_parameters(gradient, kernel.names)
        names = kernel.name_parameters_as_list
        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)
        return gradient

    @classmethod
    def evaluate_cross_cov_defined_by_params(cls, params, inputs_1, inputs_2, dimension, *args,
                                             **kernel_parameters):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are related to the parameters of the first
            kernel and so on.
        :param inputs_1: {(str) kernel_name: np.array(nxd)}
        :param inputs_2: {(str) kernel_name: np.array(kxd)}
        :param dimension: [int] list with the dimensions of the kernel
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        return kernel.cross_cov_dict(inputs_1, inputs_2)

    @classmethod
    def evaluate_cross_cov_defined_by_params_array(cls, params, inputs_1, inputs_2, dimension,
                                                   *args, **kernel_parameters):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k)) The first part are related to the parameters of the first
            kernel and so on.
        :param inputs_1: np.array(nxd)
        :param inputs_2: np.array(nxd)
        :param dimension: [int] list with the dimensions of the kernel
        :param args: [str] List with the names of the kernels.
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """

        kernel = cls.define_kernel_from_array(dimension, params, *args, **kernel_parameters)

        inputs_1 = kernel.inputs_from_array_to_dict(inputs_1)
        inputs_2 = kernel.inputs_from_array_to_dict(inputs_2)

        return kernel.cross_cov_dict(inputs_1, inputs_2)

    @staticmethod
    def compare_kernels(kernel1, kernel2):
        """
        Compare the values of kernel1 and kernel2. Returns True if they're equal, otherwise it
        return False.

        :param kernel1: ProductKernels instance object
        :param kernel2: ProductKernels instance object
        :return: boolean
        """

        if kernel1.name != kernel2.name:
            return False

        if kernel1.dimension != kernel2.dimension:
            return False

        if kernel1.dimension_parameters != kernel2.dimension_parameters:
            return False

        if kernel1.names != kernel2.names:
            return False

        for i in xrange(len(kernel1.names)):
            name1 = kernel1.names[i]

            kernel_1 = kernel1.kernels[name1]
            kernel_2 = kernel2.kernels[name1]

            if name1 == MATERN52_NAME:
                if Matern52.compare_kernels(kernel_1, kernel_2) is False:
                    return False

            if name1 == TASKS_KERNEL_NAME:
                if TasksKernel.compare_kernels(kernel_1, kernel_2) is False:
                    return False

        return True

    @staticmethod
    def parameters_from_list_to_dict(params, **kwargs):
        """
        Converts a list of parameters to dictionary using the order of the kernel.

        :param params: [float]
        :param kwargs:{
            'dimensions': [float],
            'kernels': [str],
            SAME_CORRELATION: (boolean),
        }
        :return: {
           PARAM_NAME: [float] or float
        }
        """

        parameters = {}

        for dim, kernel in zip(kwargs['dimensions'], kwargs['kernels']):
            if kernel == MATERN52_NAME:
                n_params = get_number_parameters_kernel([kernel], [dim])
                param_dict = Matern52.parameters_from_list_to_dict(params[0: n_params])
                params = params[n_params:]
                parameters.update(param_dict)
            elif kernel == TASKS_KERNEL_NAME:
                n_params = get_number_parameters_kernel([kernel], [dim], **kwargs)
                param_dict = TasksKernel.parameters_from_list_to_dict(params[0: n_params])
                params = params[n_params:]
                parameters.update(param_dict)
            elif kernel == ORNSTEIN_KERNEL:
                n_params = 2
                param_dict = Ornstein.parameters_from_list_to_dict(params[0: n_params])
                params = params[n_params:]
                parameters.update(param_dict)

        return parameters
