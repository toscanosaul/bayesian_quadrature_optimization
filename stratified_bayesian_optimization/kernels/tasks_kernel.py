from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.constant import (
    TASKS_KERNEL_NAME,
    LOWER_TRIANG_NAME,
)
from stratified_bayesian_optimization.lib.util import \
    convert_dictionary_gradient_to_simple_dictionary


class TasksKernel(AbstractKernel):

    def __init__(self, n_tasks, lower_triang):
        """

        :param n_tasks: (int) number of tasks
        :param lower_triang: (ParameterEntity) If L(i, j) = exp(lower_triang[cum_sum(i)+j]), then
            Z = L * L^T where Z[i,j] = cov(Task_i, Task_j).
        """

        name = TASKS_KERNEL_NAME
        dimension = 1

        super(TasksKernel, self).__init__(name, dimension)

        self.lower_triang = lower_triang
        self.n_tasks = n_tasks
        self.base_cov_matrix = None
        self.chol_base_cov_matrix = None
        self.number_parameters = np.cumsum(xrange(n_tasks + 1))[n_tasks]

    @property
    def hypers(self):
        return {
            self.lower_triang.name: self.lower_triang
        }

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return [(self.lower_triang.name, [(i, None) for i in xrange(self.number_parameters)])]

    def set_parameters(self, lower_triang=None):
        """

        :param lower_triang: ParameterEntity
        """

        self.base_cov_matrix = None

        if lower_triang is not None:
            self.lower_triang = lower_triang
            self.compute_cov_matrix()

    @classmethod
    def define_kernel_from_array(cls, dimension, params):
        """
        :param dimension: (int) number of tasks
        :param params: (np.array(k))

        :return: TasksKernel
        """

        lower_triang = ParameterEntity(LOWER_TRIANG_NAME, params, None)

        return cls(dimension, lower_triang)

    def cov(self, inputs):
        """

        :param inputs: np.array(nx1)
        :return: np.array(nxn)
        """
        return self.cross_cov(inputs, inputs)

    def compute_cov_matrix(self):
        """
        Compute L * L(i, j)^T from self.lower_triang
        """

        if self.base_cov_matrix is not None:
            return

        count = 0
        L = np.zeros((self.n_tasks, self.n_tasks))
        for i in range(self.n_tasks):
            for j in range(i + 1):
                L[i, j] = np.exp(self.lower_triang.value[count + j])
            count += i + 1

        covM = np.dot(L, np.transpose(L))

        self.chol_base_cov_matrix = L
        self.base_cov_matrix = covM

    def cross_cov(self, inputs_1, inputs_2):
        """

        :param inputs_1: np.array(nx1)
        :param inputs_2: np.array(mx1)
        :return: np.array(nxm)
        """

        self.compute_cov_matrix()

        s, t = np.meshgrid(inputs_1, inputs_2)
        s = s.astype(int)
        t = t.astype(int)

        cov = self.base_cov_matrix[s, t].transpose()

        return cov

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nx1)
        :return: {
            'lower_triang': {'entry (int)': np.array(nxn)}
        }
        """

        self.compute_cov_matrix()

        gradient_base_tasks = GradientTasksKernel.gradient_respect_parameters(
            self.chol_base_cov_matrix, self.n_tasks)

        gradient = {}
        gradient[self.lower_triang.name] = {}

        N = inputs.shape[0]

        for param_index in range(self.lower_triang.dimension):
            der_covariance = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1):
                    der_covariance[i, j] = \
                        gradient_base_tasks[param_index][inputs[i, 0], inputs[j, 0]]
                    der_covariance[j, i] = der_covariance[i, j]
            gradient[self.lower_triang.name][param_index] = der_covariance

        return gradient

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1x1)
        :param inputs: np.array(nx1)

        :return: np.array(nx1)
        """

        return np.zeros((inputs.shape[0], 1))

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k))
        :param inputs: np.array(nx1)
        :param dimension: (int) number of tasks
        :return: cov(inputs) where the kernel is defined with params
        """

        task_kernels = cls.define_kernel_from_array(dimension, params)
        return task_kernels.cov(inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k))
        :param inputs: np.array(nx1)
        :param dimension: (int) number of tasks
        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """
        task_kernels = cls.define_kernel_from_array(dimension, params)
        gradient = task_kernels.gradient_respect_parameters(inputs)

        names = task_kernels.name_parameters_as_list

        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)

        return gradient


class GradientTasksKernel(object):

    @staticmethod
    def gradient_respect_parameters(chol_base_cov_matrix, n_tasks):
        """
        Compute gradient of cov[i,j] respect to each element of lower_triang for each tasks i and j

        :param chol_base_cov_matrix: (np.array(n_tasks, n_tasks))
        :param n_tasks: (int)
        :return: {
            'entry (int)': np.array(number_tasks x number_tasks),
        }
        """

        gradient = {}
        count = 0
        for i in range(n_tasks):
            for j in range(i + 1):
                tmp_der = np.zeros((n_tasks, n_tasks))
                tmp_der[i, j] = chol_base_cov_matrix[i, j]
                tmp_der_mat = (np.dot(tmp_der, chol_base_cov_matrix.transpose()))
                tmp_der_mat += tmp_der_mat.transpose()
                gradient[count + j] = tmp_der_mat
            count += i + 1

        return gradient
