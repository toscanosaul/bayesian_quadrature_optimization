from __future__ import absolute_import

from collections import defaultdict

import numpy as np

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    SMALLEST_NUMBER,
    SMALLEST_POSITIVE_NUMBER,
    LARGEST_NUMBER,
)


def convert_dictionary_gradient_to_simple_dictionary(dictionary, order_keys):
    """

    :param dictionary: {
            (str) name: {(int): nxn} or nxn,
        }
    :param order_keys: ([(name_key, keys)]) (keys can be other list if we have a dictionary of
        dictionaries, otherwise keys=None)
    :return: {
        (int) : nxn
    } the indexes are set based on order_keys.
    """

    result = []

    for key in order_keys:
        if key[1] is None:
            result += [np.array(dictionary[key[0]])]
        else:
            for sub_key in key[1]:
                result += [np.array(dictionary[key[0]][sub_key[0]])]

    simple_dictionary = defaultdict()

    for index, element in enumerate(result):
        simple_dictionary[index] = element

    return simple_dictionary


def convert_dictionary_from_names_kernels_to_only_parameters(dictionary, order_kernels):
    """

    :param dictionary: {
            (str) kernel: {keys_kernel: value_kernel},
        }
    :param order_kernels: [str]
    :return: {
        {keys_kernel + 'kernel_name': value_kernel}
    } the indexes are set based on order_keys.
    """
    result = {}
    for kernel in order_kernels:
        for parameter in dictionary[kernel]:
            result[parameter] = dictionary[kernel][parameter]
    return result


def separate_numpy_arrays_in_lists(array, division):
    """
    Separate the m-axis of the array in a list such that [array(nxdivision), array(nx(division:))]
    :param array: np.array(nxm) or np.array(n)
    :param division: (int)
    :return: [np.array(nxdivision), np.array(nx(division:))]
    """
    if len(array.shape) == 2:
        return [array[:, 0: division], array[:, division: array.shape[1]]]
    else:
        return [array[0: division], array[division: array.shape[1]]]

def wrapper_fit_gp_regression(self):
    """
    Wrapper of fit_gp_regression
    :param self: instance of class GPFittingGaussian
    :return: updated self
    """

    return self.fit_gp_regression()

def get_number_parameters_kernel(kernel_name, dim):
    """
    Returns the number of parameters associated to the kernel.

    :param kernel_name: [str]
    :param dim: [int]
    :return: int
    """

    if kernel_name[0] == MATERN52_NAME:
        return dim[0] + 1

    if kernel_name[0] == TASKS_KERNEL_NAME:
        return np.cumsum(xrange(dim + 1))[dim[0]]

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        n_params = 0
        for name, dimension in zip(kernel_name[1:], dim[1:]):
            n_params += get_number_parameters_kernel(name, dimension)
        return n_params

    raise NameError(kernel_name + " doesn't exist")

def get_default_values_kernel(kernel_name, dim):
    """
    Returns default values for the kernel_name.

    :param kernel_name: [str]
    :param dim: [int]
    :return: [float]
    """

    if kernel_name[0] == MATERN52_NAME:
        return  list(np.ones(get_number_parameters_kernel(kernel_name, dim)))

    if kernel_name[0] == TASKS_KERNEL_NAME:
        return list(np.zeros(get_number_parameters_kernel(kernel_name, dim)))

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        values = []
        for name, dimension in zip(kernel_name[1:], dim[1:]):
            values += get_default_values_kernel(name, dimension)
        return values
