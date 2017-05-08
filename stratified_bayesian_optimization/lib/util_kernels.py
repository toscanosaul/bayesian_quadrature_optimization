from __future__ import absolute_import

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
)
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel


def find_define_kernel_from_array(kernel_name):
    """

    :param kernel_name: (str) Name of the kernel
    :return: define_kernel_from_array associated to kernel_name
    """

    if kernel_name == MATERN52_NAME:
        return Matern52.define_kernel_from_array

    if kernel_name == TASKS_KERNEL_NAME:
        return TasksKernel.define_kernel_from_array

    raise NameError(kernel_name + " doesn't exist")

def find_kernel_constructor(kernel_name):
    """

    :param kernel_name: (str) Name of the kernel
    :return: kernel constructor
    """

    if kernel_name == MATERN52_NAME:
        return Matern52

    if kernel_name == TASKS_KERNEL_NAME:
        return TasksKernel

    raise NameError(kernel_name + " doesn't exist")
