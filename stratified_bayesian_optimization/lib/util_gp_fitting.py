from __future__ import absolute_import

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels


def get_kernel_default(kernel_name, dimension):
    """
    Returns a default kernel object associated to the kernel_name
    :param kernel_name: [(str)]
    :param dimension: [int]
    :return: kernel object
    """
    if kernel_name[0] == MATERN52_NAME:
        return Matern52.define_default_kernel(dimension[0])
    if kernel_name[0] == TASKS_KERNEL_NAME:
        return TasksKernel.define_default_kernel(dimension[0])
    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        return ProductKernels.define_default_kernel(dimension, kernel_name[1, :])


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
