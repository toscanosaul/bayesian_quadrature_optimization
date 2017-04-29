from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel


class TasksKernel(AbstractKernel):

    def __init__(self, n_tasks, lower_triang):
        """

        :param n_tasks: (int) number of tasks
        :param lower_triang: (ParameterEntity) If L(i, j) = exp(lower_triang[cum_sum(i)+j]), then
            Z = L * L^T where Z[i,j] = cov(Task_i, Task_j).
        """

        name = 'Tasks_Kernel'
        dimension = 1

        super(TasksKernel, self).__init__(name, dimension)

        self.lower_triang = lower_triang
        self.n_tasks = n_tasks
        self.base_cov_matrix = None
        self.chol_base_cov_matrix = None

    @property
    def hypers(self):
        return {
            self.lower_triang.name: self.lower_triang
        }

    def set_parameters(self, lower_triang=None):
        """

        :param lower_triang: ParameterEntity
        """

        self.base_cov_matrix = None

        if lower_triang is not None:
            self.lower_triang = lower_triang
            self.compute_cov_matrix()

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

        cov = self.base_cov_matrix[s,t].transpose()

        return cov

    def gradient_respect_parameters(self, inputs):
        """

        :param inputs: np.array(nx1)
        :return: {
            'entry (int)': np.array(nxn),
        }
        """

        gradient_base_tasks = GradientTasksKernel.gradient_respect_parameters(
            self.chol_base_cov_matrix, self.n_tasks)

        gradient = {}

        N = inputs.shape[0]

        for param_index in range(self.lower_triang.dimension):
            der_covariance = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1):
                    der_covariance[i, j] = \
                        gradient_base_tasks[param_index][inputs[i,0], inputs[j,0]]
                    der_covariance[j, i] = der_covariance[i, j]
            gradient[param_index] = der_covariance

        return gradient

    def grad_respect_point(self, point, inputs):
        """
        Computes the vector of the gradients of cov(point, inputs) respect point.

        :param point: np.array(1x1)
        :param inputs: np.array(nx1)

        :return: np.array(nx1)
        """

        return np.zeros((inputs.shape[0], 1))


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
