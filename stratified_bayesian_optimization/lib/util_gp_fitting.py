from __future__ import absolute_import

from stratified_bayesian_optimization.lib.constant import(
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    SCALED_KERNEL,
    SIGMA2_NAME,
    LENGTH_SCALE_NAME,
    LOWER_TRIANG_NAME,
)
from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
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

    if kernel_name[0] == SCALED_KERNEL:
        if kernel_name[1] == MATERN52_NAME:
            return ScaledKernel.define_default_kernel(dimension[0], bounds, default_values,
                                                      parameters_priors, *([MATERN52_NAME], ))

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
    if kernel_name == SCALED_KERNEL:
        return ScaledKernel


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

    if type_kernels[0] == SCALED_KERNEL:
        kwargs['dimensions'] = dimensions[0:]
        kwargs['kernels'] = type_kernels[1]

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


def define_prior_parameters_using_data(data, type_kernel, dimensions, sigma2=None):
    """
    Defines value of the parameters of the prior distributions of the kernel's parameters.

    :param data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
    :param type_kernel: [str]
    :param dimensions: [int], It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
    :param: sigma2: float
    :return: {
        SIGMA2_NAME: float,
        LENGTH_SCALE_NAME: [float],
        LOWER_TRIANG_NAME: [float],
    }
    """

    # We assume that there is at most one task kernel, and mattern52 kernel in the product.

    parameters_priors = {
        SIGMA2_NAME: None,
        LENGTH_SCALE_NAME: None,
        LOWER_TRIANG_NAME: None,
    }

    index = -1

    if TASKS_KERNEL_NAME in type_kernel:
        index = type_kernel.index(TASKS_KERNEL_NAME)
        index_tasks = 0
        for i in xrange(1, index):
            index_tasks += dimensions[i]
        n_tasks = dimensions[index]
        tasks_index = data['points'][:, index_tasks]
        task_data = data.copy()
        task_data['points'] = tasks_index.reshape((len(tasks_index), 1))
        task_parameters = TasksKernel.define_prior_parameters(task_data, n_tasks)
        parameters_priors[LOWER_TRIANG_NAME] = task_parameters[LOWER_TRIANG_NAME]

    if MATERN52_NAME in type_kernel:
        m = data['points'].shape[1]
        indexes = [i for i in range(m) if i != index - 1]
        points_matern = data['points'][:, indexes]
        matern_data = data.copy()
        matern_data['points'] = points_matern
        matern52_parameters = Matern52.define_prior_parameters(matern_data, len(indexes))
        parameters_priors[LENGTH_SCALE_NAME] = matern52_parameters[LENGTH_SCALE_NAME]

        if SCALED_KERNEL in type_kernel:
            parameters = \
                ScaledKernel.define_prior_parameters(data, len(indexes), var_evaluations=sigma2)
            parameters_priors[SIGMA2_NAME] = parameters[SIGMA2_NAME]

    return parameters_priors
