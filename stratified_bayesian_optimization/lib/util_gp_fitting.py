from __future__ import absolute_import

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
    combine_vectors,
)


def get_kernel_default(kernel_name, dimension, bounds=None, default_values=None,
                       parameters_priors=None):
    """
    Returns a default kernel object associated to the kernel_name
    :param kernel_name: [str]
    :param dimension: [int]. It's the number of tasks for the task kernel.
    :param bounds: [[float, float]], lower bound and upper bound for each entry. This parameter
            is to compute priors in a smart way.
    :param default_values: np.array(k), default values for the parameters of the kernel
    :param parameters_priors: {
            SIGMA2_NAME: float,
            LENGTH_SCALE_NAME: [float],
            LOWER_TRIANG_NAME: [float],
        }

    :return: kernel object
    """
    if kernel_name[0] == MATERN52_NAME:
        return Matern52.define_default_kernel(dimension[0], bounds, default_values,
                                              parameters_priors)
    if kernel_name[0] == TASKS_KERNEL_NAME:
        return TasksKernel.define_default_kernel(dimension[0], bounds, default_values,
                                                 parameters_priors)
    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        values = []
        cont = 0
        bounds_ = []
        cont_b = 0
        for name, dim in zip(kernel_name[1:], dimension[1:]):
            n_params = get_number_parameters_kernel([name], [dim])
            if default_values is not None:
                value_kernel = default_values[cont: cont + n_params]
            else:
                value_kernel = None

            if bounds is not None:
                if name == MATERN52_NAME:
                    bounds_.append(bounds[cont_b: cont_b + dim])
                    cont_b += dim
                if name == TASKS_KERNEL_NAME:
                    bounds_.append(bounds[cont_b: cont_b + 1])
                    cont_b += 1
            cont += n_params
            values.append(value_kernel)

        if len(bounds_) > 0:
            bounds = bounds_

        return ProductKernels.define_default_kernel(dimension[1:], bounds, values,
                                                    parameters_priors, kernel_name[1:])


def get_kernel_class(kernel_name):
    """
    Returns the kernel class associated to the kernel_name
    :param kernel_name: (str)
    :return: class
    """

    if kernel_name == MATERN52_NAME:
        return Matern52
    if kernel_name == TASKS_KERNEL_NAME:
        return TasksKernel
    if kernel_name == PRODUCT_KERNELS_SEPARABLE:
        return ProductKernels


def parameters_kernel_from_list_to_dict(params, type_kernels, dimensions):
    """
    Converts a list of parameters to dictionary using the order of the kernel.

    :param params: [float]
    :param type_kernels: [str]
    :param dimensions: [float]

    :return: {
       PARAM_NAME: [float] or float
    }
    """

    kernel = get_kernel_class(type_kernels[0])

    kwargs = {}
    if type_kernels[0] == PRODUCT_KERNELS_SEPARABLE:
        kwargs['dimensions'] = dimensions[1:]
        kwargs['kernels'] = type_kernels[1:]

    return kernel.parameters_from_list_to_dict(params, **kwargs)

def wrapper_log_prob(vector, self):
    """
    Wrapper of log_prob

    :param vector: (np.array(n)) The order is defined in the function get_parameters_model
            of the class model.
    :param self: instance of class GPFittingGaussian

    :return: float
    """

    return self.log_prob_parameters(vector)
