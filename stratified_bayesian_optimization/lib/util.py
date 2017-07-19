from __future__ import absolute_import

from collections import defaultdict

import numpy as np

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    LENGTH_SCALE_NAME,
    SIGMA2_NAME,
    LOWER_TRIANG_NAME,
    SCALED_KERNEL,
    SAME_CORRELATION,
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


def wrapper_fit_gp_regression(self, **kwargs):
    """
    Wrapper of fit_gp_regression
    :param self: instance of class GPFittingGaussian
    :param kwargs:
        - 'start': (np.array(n)) starting point of the optimization of the llh.
        - 'random_seed': int
    :return: updated self
    """

    return self.fit_gp_regression(**kwargs)


def wrapper_evaluate_objective_function(point, cls_, name_module, n_samples):
    """
    Wrapper of evaluate_function in training_data
    :param cls: TrainingDataService
    :param name_module: (str) Name of the module of the problem
    :param point: [float]
    :param n_samples: int. If noise is true, we take n_samples of the function to estimate its
        value.
    :return: float
    """

    module = __import__(name_module, globals(), locals(), -1)

    return cls_.evaluate_function(module, point, n_samples)


def get_number_parameters_kernel(kernel_name, dim, **kernel_parameters):
    """
    Returns the number of parameters associated to the kernel.

    :param kernel_name: [str]
    :param dim: [int], for standard kernels the list consists of only one element. For the product
        of kernels the lists consists of the dimension of the product of kernels, and each of the
        kernels of the product.
    :param kernel_parameters: additional kernel parameters,
        - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.
    :return: int
    """

    if kernel_name[0] == MATERN52_NAME:
        return dim[0]

    if kernel_name[0] == TASKS_KERNEL_NAME:
        same_correlation = kernel_parameters.get(SAME_CORRELATION, False)

        if not same_correlation:
            return np.cumsum(xrange(dim[0] + 1))[dim[0]]
        else:
            return min(dim[0], 2)

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        n_params = 0
        for name, dimension in zip(kernel_name[1:], dim[1:]):
            n_params += get_number_parameters_kernel([name], [dimension], **kernel_parameters)
        return n_params

    raise NameError(kernel_name[0] + " doesn't exist")


def get_default_values_kernel(kernel_name, dim, same_correlation=False, **parameters_priors):
    """
    Returns default values for the kernel_name.

    :param kernel_name: [str]
    :param dim: [int]
    :param same_correlation: (boolean) Parameter for the task kernel
    :param parameters_priors:
            -SIGMA2_NAME: float,
            -LENGTH_SCALE_NAME: [float],
            -LOWER_TRIANG_NAME: [float],
    :return: [float]
    """

    if kernel_name[0] == SCALED_KERNEL:
        sigma2 = [parameters_priors.get(SIGMA2_NAME, 1.0)]
        if kernel_name[1] == MATERN52_NAME:
            ls = parameters_priors.get(LENGTH_SCALE_NAME, dim[0] * [1.0])
            return ls + sigma2

    if kernel_name[0] == MATERN52_NAME:
        ls = parameters_priors.get(LENGTH_SCALE_NAME, dim[0] * [1.0])
        return ls

    if kernel_name[0] == TASKS_KERNEL_NAME:
        n_params = get_number_parameters_kernel(kernel_name, dim,
                                                **{SAME_CORRELATION: same_correlation})
        tasks_kernel_chol = parameters_priors.get(LOWER_TRIANG_NAME, n_params * [0.0])
        return tasks_kernel_chol

    if kernel_name[0] == PRODUCT_KERNELS_SEPARABLE:
        values = []

        for name, dimension in zip(kernel_name[1:], dim[1:]):
            values += get_default_values_kernel([name], [dimension], **parameters_priors)

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


def combine_vectors(vector1, vector2, indexes1):
    """
    Combine vector1 and vector2 into one vector v where
    v[indexes1[i]] = vector1[i], and
    v[indexes2[i]] = vector2[i] where indexes2 = np.arange(dimension) - indexes1

    :param vector1: np.array(n1)
    :param vector2: np.array(n2)
    :param indexes1: [int]

    :return: np.array(n1+n2)
    """

    dimension = len(vector1) + len(vector2)
    vector = np.zeros(dimension)

    for index, index1 in enumerate(indexes1):
        vector[index1] = vector1[index]

    indexes2 = [i for i in list(np.arange(dimension)) if i not in indexes1]
    for index, index2 in enumerate(indexes2):
        vector[index2] = vector2[index]

    return vector


def separate_vector(vector, indexes1):
    """
    Separate vector into two vectors vector1, vector2 where
    vector1 = vector[indexes1],
    vector2 = vector[indexes1^c]

    :param vector: np.array(n)
    :param indexes1: [int]
    :return: [np.array(n1), np.array(n2)] where n = n1 + n2
    """
    dimension = len(vector)

    vector1 = vector[indexes1]

    indexes2 = [i for i in list(np.arange(dimension)) if i not in indexes1]
    vector2 = vector[indexes2]

    return [vector1, vector2]


def wrapper_optimization(start, *args):
    """
    Wrapper of optimization.optimize


    :param start: np.array(n), starting point of the optimization
    :param args: args[0] is an optimization instance, and args[1:] are arrguments to pass to
        the objective function and its gradient.

    :return: {
        'solution': np.array(n),
        'optimal_value': float,
        'gradient': np.array(n),
        'warnflag': int,
        'task': str
    }
    """

    return args[0].optimize(start, *args[1:])


def wrapper_objective_voi(point, self):
    """
    Wrapper of objective_voi
    :param self: instance of the acquisition function
    :param point: np.array(n)

    :return: float
    """
    return self.objective_voi(point)


def wrapper_gradient_voi(point, self):
    """
    Wrapper of objective_voi (an acquisition function)
    :param self: instance of the acquisition function
    :param point: np.array(n)

    :return: np.array(n)
    """
    return self.grad_obj_voi(point)

def wrapper_evaluate_quadrature_cross_cov(point, historical_points, parameters_kernel, self):
    """
    Wrapper of evaluate quadrature cross cov

    :param point: np.array(1xn)
    :param historical_points: np.array(mxk)
    :param parameters_kernel: np.array(l)
    :param self: instance of bayesian quadrature
    :return: np.array(1xm)
    """

    return self.evaluate_quadrature_cross_cov(
                point, historical_points, parameters_kernel)


def wrapper_compute_vector_b(point, compute_vec_covs, compute_b_new, historical_points,
                             parameters_kernel, candidate_point, self):
    """
    Wrapper of the function that computes B(x, i) and B(new, i)

    :param point: np.array(1xn)
    :param compute_vec_covs: boolean
    :param compute_b_new: boolean
    :param historical_points: np.array(kxm)
    :param parameters_kernel: np.array(l)
    :param candidate_point: np.array(1xm)
    :param self: bq instance
    :return: {
        'b_new': float,
        'vec_covs': np.array(k),
    }
    """
    b_new = None
    vec_covs = None

    if compute_vec_covs:
        vec_covs = self.evaluate_quadrature_cross_cov(
            point, historical_points, parameters_kernel)
    if compute_b_new:
        b_new = self.evaluate_quadrature_cross_cov(
            point, candidate_point, parameters_kernel)

    return {
        'b_new': b_new,
        'vec_covs': vec_covs,
    }


def wrapper_evaluate_sbo(point, task, self):
    """

    :param point: np.array(1xn)
    :param task: (int)
    :param self: sbo instance
    :return: float
    """
    point = np.concatenate((point, np.array([[task]])), axis=1)
    return self.evaluate(point)