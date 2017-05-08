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
)


def get_kernel_default(kernel_name, dimension, default_values=None):
    """
    Returns a default kernel object associated to the kernel_name
    :param kernel_name: [str]
    :param dimension: [int]
    :param default_values: np.array(k), default values for the parameters of the kernel
    :return: kernel object
    """
    if kernel_name[0] == MATERN52_NAME:
        return Matern52.define_default_kernel(dimension[0], default_values)
    if kernel_name[0] == TASKS_KERNEL_NAME:
        return TasksKernel.define_default_kernel(dimension[0], default_values)
    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        values = []
        cont = 0
        for name, dimension in zip(kernel_name[1:], dimension):
            n_params = get_number_parameters_kernel(name, dimension)
            value_kernel = default_values[cont: cont + n_params]
            cont += n_params
            values.append(value_kernel)

        return ProductKernels.define_default_kernel(dimension, values, kernel_name[1:])


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
