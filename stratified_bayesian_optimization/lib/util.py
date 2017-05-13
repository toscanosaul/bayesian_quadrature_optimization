from __future__ import absolute_import

from collections import defaultdict

import numpy as np

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
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
        return [array[0: division], array[division: len(array)]]

def wrapper_fit_gp_regression(self):
    """
    Wrapper of fit_gp_regression
    :param self: instance of class GPFittingGaussian
    :return: updated self
    """

    return self.fit_gp_regression()

def wrapper_evaluate_objective_function(cls, point, module, n_samples):
    """
    Wrapper of evaluate_function in training_data
    :param cls: TrainingDataService
    :param module:
    :param point: [float]
    :param n_samples: int. If noise is true, we take n_samples of the function to estimate its
        value.
    :return: float
    """

    return cls.evaluate_function(module, point, n_samples)

def get_number_parameters_kernel(kernel_name, dim):
    """
    Returns the number of parameters associated to the kernel.

    :param kernel_name: [str]
    :param dim: [int], for standard kernels the list consists of only one element. For the product
        of kernels the lists consists of the dimension of the product of kernels, and each of the
        kernels of the product.
    :return: int
    """

    if kernel_name[0] == MATERN52_NAME:
        return dim[0] + 1

    if kernel_name[0] == TASKS_KERNEL_NAME:
        return np.cumsum(xrange(dim[0] + 1))[dim[0]]

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        n_params = 0
        for name, dimension in zip(kernel_name[1:], dim[1:]):
            n_params += get_number_parameters_kernel([name], [dimension])
        return n_params

    raise NameError(kernel_name[0] + " doesn't exist")

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
            values += get_default_values_kernel([name], [dimension])
        return values

def convert_list_to_dictionary(ls):
    """

    :param ls: [object]
    :return: {i: ls[i]}
    """

    d = {}
    for index, value in enumerate(ls):
        d[index] = value

    return d

def convert_dictionary_to_list(dictionary):
    """
    Return list where list[i] = b iff dictionary[i] = b.

    :param dictionary: {(int) i: object}
    :return: [object]
    """
    ls = [None] * len(dictionary)

    for key, value in dictionary.iteritems():
        ls[key] = value

    return ls

def expand_dimension_vector(x, change_indexes, default_x):
    """
    Expand the dimension of x, where the new_x[i] = x[index] iff i is in
    change_indexes, otherwise new_x[i] = default_x[index].

    :param x: np.array(n)
    :param change_indexes: [int], where its length is less than n.
    :param default_x: np.array(m)

    :return: np.array(m)
    """

    index = 0
    new_x = default_x.copy()
    for j in change_indexes:
        new_x[j] = x[index]
        index += 1
    return new_x

def reduce_dimension_vector(x, change_indexes):
    """
    Reduce the dimension of the vector, where the entries in change_indexes are conserved.

    :param x: np.array(n)
    :param change_indexes: [int]

    :return: np.array(m)
    """

    new_x = np.zeros(len(change_indexes))
    index = 0
    for j in change_indexes:
        new_x[index] = x[j]
        index += 1
    return new_x
