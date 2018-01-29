from __future__ import absolute_import

import numpy as np

from copy import deepcopy

from stratified_bayesian_optimization.kernels.abstract_kernel import AbstractKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.constant import (
    TASKS_KERNEL_NAME,
    LOWER_TRIANG_NAME,
    SMALLEST_POSITIVE_NUMBER,
    SAME_CORRELATION,
)
from stratified_bayesian_optimization.lib.util import (
    get_number_parameters_kernel,
    convert_dictionary_gradient_to_simple_dictionary,
)
from stratified_bayesian_optimization.priors.gaussian import GaussianPrior
from stratified_bayesian_optimization.priors.multivariate_normal import MultivariateNormalPrior
from stratified_bayesian_optimization.priors.log_normal_square import LogNormalSquare


class TasksKernel(AbstractKernel):

    def __init__(self, n_tasks, lower_triang, same_correlation=False, **kernel_parameters):
        """

        :param n_tasks: (int) number of tasks
        :param lower_triang: (ParameterEntity) If L(i, j) = exp(lower_triang[cum_sum(i)+j]), then
            Z = L * L^T where Z[i,j] = cov(Task_i, Task_j).
            If same_correlation is True, then
        :param same_correlation: (boolena) If True, it uses the same correlation for all tasks.
            We then have two parameters in total: var(task_i, task_i) and cov(task_i, task_j).
            In that case, the lower_triang consists of only the log(r) and log(covariance), where
            variance = covariance * (n_tasks - 1) + r (this guarantees that the matrix is P.D.)
        """

        name = TASKS_KERNEL_NAME
        dimension = 1

        if not same_correlation:
            dimension_parameters = get_number_parameters_kernel([name], [n_tasks])
        else:
            dimension_parameters = min(n_tasks, 2)

        super(TasksKernel, self).__init__(name, dimension, dimension_parameters)

        self.same_correlation = same_correlation
        self.lower_triang = lower_triang
        self.n_tasks = n_tasks
        self.base_cov_matrix = None
        self.chol_base_cov_matrix = None

    @property
    def hypers(self):
        return {
            self.lower_triang.name: self.lower_triang
        }

    @property
    def hypers_as_list(self):
        """
        This function defines the default order of the parameters.
        :return: [ParameterEntity]
        """
        return [self.lower_triang]

    @property
    def hypers_values_as_array(self):
        """

        :return: np.array(n)
        """
        return self.lower_triang.value

    def sample_parameters(self, number_samples, random_seed=None):
        """

        :param number_samples: (int) number of samples
        :param random_seed: (int)
        :return: np.array(number_samples x k)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        parameters = self.hypers
        return parameters[self.lower_triang.name].sample_from_prior(number_samples)

    def get_bounds_parameters(self):
        """
        Return bounds of the parameters of the kernel
        :return: [(float, float)]
        """
        return self.lower_triang.bounds

    @property
    def name_parameters_as_list(self):
        """

        :return: ([(name_param, name_params)]) name_params can be other list if name_param
            represents several parameters (like an array), otherwise name_params=None.
        """
        return [(self.lower_triang.name, [(i, None) for i in range(self.dimension_parameters)])]

    def set_parameters(self, lower_triang=None):
        """

        :param lower_triang: ParameterEntity
        """

        self.base_cov_matrix = None

        if lower_triang is not None:
            self.lower_triang = lower_triang
            self.compute_cov_matrix()

    def update_value_parameters(self, params):
        """

        :param params: np.array(n)
        """
        self.base_cov_matrix = None
        self.lower_triang.set_value(params)
        self.compute_cov_matrix()

    @classmethod
    def define_kernel_from_array(cls, dimension, params,  **kwargs):
        """
        :param dimension: (int) number of tasks
        :param params: (np.array(k))
        :param kwargs: {SAME_CORRELATION: boolean}

        :return: TasksKernel
        """

        lower_triang = ParameterEntity(LOWER_TRIANG_NAME, params, None)

        same_correlation = kwargs.get(SAME_CORRELATION, False)

        return cls(dimension, lower_triang, same_correlation=same_correlation)

    @classmethod
    def define_default_kernel(cls, dimension, bounds=None, default_values=None,
                              parameters_priors=None, **kwargs):
        """
        :param dimension: (int) Number of tasks.
        :param bounds: [[float, float]], lower bound and upper bound for each entry. This parameter
                is to compute priors in a smart way.
        :param default_values: np.array(k)
        :param parameters_priors: {
                        LOWER_TRIANG_NAME: [float]
                    }
        :param kwargs: {SAME_CORRELATION: boolean}

        :return: TasksKernel
        """

        same_correlation = kwargs.get(SAME_CORRELATION, False)

        if not same_correlation:
            n_params = get_number_parameters_kernel([TASKS_KERNEL_NAME], [dimension])
        else:
            n_params = min(dimension, 2)

        if parameters_priors is None:
            parameters_priors = {}


        if default_values is None:
            tasks_kernel_chol = parameters_priors.get(LOWER_TRIANG_NAME, n_params * [0.0])
            default_values = np.array(tasks_kernel_chol)

        kernel = TasksKernel.define_kernel_from_array(
            dimension, default_values, **kwargs)

        if dimension == 1:
            kernel.lower_triang.prior = LogNormalSquare(1, 1.0, np.sqrt(default_values[0]))
            kernel.lower_triang.bounds = [(SMALLEST_POSITIVE_NUMBER, None)]
        else:
            cov = np.eye(n_params)
            kernel.lower_triang.prior = MultivariateNormalPrior(n_params, default_values, cov)


        return kernel

    def cov(self, inputs):
        """

        :param inputs: np.array(nx1)
        :return: np.array(nxn)
        """
        return self.cross_cov(inputs, inputs)

    def compute_cov_matrix(self):
        """
        Compute L * L(i, j)^T from self.lower_triang if self.same_correlation is False.
        Otherwise, cov is builded by assumin same covariance for all tasks.
        """

        if self.base_cov_matrix is not None:
            return

        if self.n_tasks == 1:
            covM = np.zeros((self.n_tasks, self.n_tasks))
            covM[0, 0] = self.lower_triang.value[0]
            L = covM
            self.chol_base_cov_matrix = L
            self.base_cov_matrix = covM

        if not self.same_correlation:
            count = 0
            L = np.zeros((self.n_tasks, self.n_tasks))
            for i in range(self.n_tasks):
                for j in range(i + 1):
                    L[i, j] = np.exp(self.lower_triang.value[count + j])
                count += i + 1

            covM = np.dot(L, np.transpose(L))
        else:
            covM = np.zeros((self.n_tasks, self.n_tasks))

            if self.n_tasks > 1:
                value = self.lower_triang.value[1]
                covM.fill(np.exp(value))

                for i in xrange(self.n_tasks):
                    covM[i, i] = np.exp(self.lower_triang.value[0]) + np.exp(value) * \
                                                                      (self.n_tasks - 1 )
            else:
                covM[0, 0] = np.exp(self.lower_triang.value[0])
            L = covM


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
        inputs = inputs.astype(int)
        self.compute_cov_matrix()

        gradient = {}
        gradient[self.lower_triang.name] = {}

        gradient_base_tasks = GradientTasksKernel.gradient_respect_parameters(
            self.chol_base_cov_matrix, self.n_tasks, self.same_correlation)

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

    def hessian_respect_point(self, point, inputs):
        """
        Computes the hessians of cov(point, inputs) respect point

        :param point:
        :param inputs:
        :return: np.array(nx1x1)
        """

        hessian = np.zeros((inputs.shape[0], 1, 1))

        return hessian

    @classmethod
    def evaluate_grad_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the gradient of the kernel defined by params respect to the point.

        :param params: np.array(k)
        :param point: np.array(1x1)
        :param inputs: np.array(nx1)
        :param dimension: (int) number of tasks
        :return: np.array(nx1)

        """
        kernel = cls.define_kernel_from_array(dimension, params)
        return kernel.grad_respect_point(point, inputs)

    @classmethod
    def evaluate_cov_defined_by_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k))
        :param inputs: np.array(nx1)
        :param dimension: (int) number of tasks
        :param kwargs: {SAME_CORRELATION: boolean}

        :return: cov(inputs) where the kernel is defined with params
        """
        same_correlation = kwargs.get(SAME_CORRELATION, False)
        task_kernels = cls.define_kernel_from_array(dimension, params,
                                                    same_correlation=same_correlation)
        return task_kernels.cov(inputs)

    @classmethod
    def evaluate_cross_cov_defined_by_params(cls, params, inputs_1, inputs_2, dimension, **kwargs):
        """
        Evaluate the covariance of the kernel defined by params.

        :param params: (np.array(k))
        :param inputs_1: np.array(nxm)
        :param inputs_2: np.array(kxm)
        :param dimension: (int) dimension of the domain of the kernel
        :param kwargs: {SAME_CORRELATION: boolean}

        :return: (np.array(nxk)) cov(inputs_1, inputs_2) where the kernel is defined with params
        """
        same_correlation = kwargs.get(SAME_CORRELATION, False)
        task_kernels = cls.define_kernel_from_array(dimension, params,
                                                    same_correlation=same_correlation)

        return task_kernels.cross_cov(inputs_1, inputs_2)

    @classmethod
    def evaluate_hessian_respect_point(cls, params, point, inputs, dimension):
        """
        Evaluate the hessian of the kernel defined by params respect to the point.

        :param params:
        :param point:
        :param inputs:
        :param dimension:
        :return:
        """
        kernel = cls.define_kernel_from_array(dimension, params)
        return kernel.hessian_respect_point(point, inputs)

    @classmethod
    def evaluate_grad_defined_by_params_respect_params(cls, params, inputs, dimension, **kwargs):
        """
        Evaluate the gradient respect the parameters of the kernel defined by params.

        :param params: (np.array(k))
        :param inputs: np.array(nx1)
        :param dimension: (int) number of tasks
        :param kwargs: {SAME_CORRELATION: boolean}
        :return: {
            (int) i: (nxn), derivative respect to the ith parameter
        }
        """
        same_correlation = kwargs.get(SAME_CORRELATION, False)
        task_kernels = cls.define_kernel_from_array(dimension, params,
                                                    same_correlation=same_correlation)
        gradient = task_kernels.gradient_respect_parameters(inputs)

        names = task_kernels.name_parameters_as_list

        gradient = convert_dictionary_gradient_to_simple_dictionary(gradient, names)

        return gradient

    @staticmethod
    def define_prior_parameters(data, dimension, same_correlation=False, var_evaluations=None):
        """
        Defines value of the parameters of the prior distributions of the kernel's parameters.

        :param data: {'points': np.array(nx1), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}. Each point is the is an index of the task.
        :param dimension: int, number of tasks
        :param same_correlation: boolean
        :return:  {
            LOWER_TRIANG_NAME: [float],
        }
        """

        if dimension == 1:
            return {LOWER_TRIANG_NAME: [var_evaluations]}

        tasks_index = data['points'][:, 0]
        data_by_tasks = {}
        for i in xrange(dimension):
            index_task = np.where(tasks_index == i)[0]
            if len(index_task) > 0:
                data_by_tasks[i] = [data['evaluations'][index_task],
                                    np.mean(data['evaluations'][index_task])]
            else:
                data_by_tasks[i] = [[]]

        # Can we include the variance of noisy evaluations in a smart way to get better estimators?

        cov_estimate = np.zeros((dimension, dimension))

        for i in xrange(dimension):
            for j in xrange(i + 1):
                a1 = len(data_by_tasks[i][0])
                a2 = len(data_by_tasks[j][0])
                d = min(a1, a2)
                if d <= 1:
                    if i == j:
                        if not same_correlation:
                            cov_estimate[i, j] = 1.0
                        else:
                            cov_estimate[i, j] = var_evaluations
                    else:
                        if not same_correlation:
                            cov_estimate[i, j] = 0.0
                            cov_estimate[j, i] = 0.0
                        else:
                            n_eval = len(data['evaluations'])
                            z = data['evaluations'][0: n_eval/2]
                            z = z - np.mean(z)

                            z_2 = data['evaluations'][n_eval / 2: n_eval]
                            z_2 = z_2 - np.mean(z_2)

                            cov = [z1 * z2 for z1 in z for z2 in z_2]
                            cov = np.mean(cov)
                            cov_estimate[i, j] = cov
                            cov_estimate[j, i] = cov_estimate[i, j]
                else:
                    mu1 = data_by_tasks[i][1]
                    mu2 = data_by_tasks[j][1]
                    a = data_by_tasks[i][0][0:d]
                    b = data_by_tasks[j][0][0:d]
                    cov_estimate[i, j] = np.sum((a - mu1) * (b - mu2)) / (d - 1.0)
                    cov_estimate[j, i] = cov_estimate[i, j]

        if same_correlation:
            var = [cov_estimate[i, i] for i in xrange(dimension)]
            task_params = []
            task_params.append(np.log(max(np.mean(var), 0.1)))

            if dimension == 1:
                return {LOWER_TRIANG_NAME: task_params}

            cov = [cov_estimate[i, j] for i in xrange(dimension) for j in xrange(dimension) if
                   i != j]
            task_params.append(np.log(max(np.mean(cov), 0.1)))

            return {LOWER_TRIANG_NAME: task_params}

        l_params = {}
        for j in range(dimension):
            for i in range(j, dimension):
                if i == j:
                    value = np.sqrt(
                        max(cov_estimate[i, j] -
                            np.sum(np.array([l_params[(i, h)] for h in xrange(i)]) ** 2),
                            SMALLEST_POSITIVE_NUMBER))
                    l_params[(i, j)] = value
                    continue
                ls_val = np.sum(
                    [l_params[(i, h)] * l_params[(j, h)] for h in xrange(min(i, j))])
                d = min(i, j)
                value = (cov_estimate[(i, j)] - ls_val) / l_params[(d, d)]
                l_params[(i, j)] = value

        task_params = []
        for i in xrange(dimension):
            for j in xrange(i + 1):
                value = l_params[(i, j)]
                task_params.append(np.log(max(value, 0.0001)))

        return {LOWER_TRIANG_NAME: task_params}

    @staticmethod
    def compare_kernels(kernel1, kernel2):
        """
        Compare the values of kernel1 and kernel2. Returns True if they're equal, otherwise it
        return False.

        :param kernel1: TasksKernel instance object
        :param kernel2: TasksKernel instance object
        :return: boolean
        """
        if kernel1.name != kernel2.name:
            return False

        if kernel1.dimension != kernel2.dimension:
            return False

        if kernel1.dimension_parameters != kernel2.dimension_parameters:
            return False

        if kernel1.n_tasks != kernel2.n_tasks:
            return False

        if np.any(kernel1.lower_triang.value != kernel2.lower_triang.value):
            return False

        if np.any(kernel1.base_cov_matrix != kernel2.base_cov_matrix):
            return False

        if np.any(kernel1.chol_base_cov_matrix != kernel2.chol_base_cov_matrix):
            return False

        if kernel1.same_correlation != kernel2.same_correlation:
            return False

        return True

    @staticmethod
    def parameters_from_list_to_dict(params, **kwargs):
        """
        Converts a list of parameters to dictionary using the order of the kernel.

        :param params: [float]

        :return: {
            LOWER_TRIANG_NAME: [float],
        }
        """

        parameters = {}
        parameters[LOWER_TRIANG_NAME] = params

        return parameters


class GradientTasksKernel(object):

    @staticmethod
    def gradient_respect_parameters(chol_base_cov_matrix, n_tasks, same_correlation=False):
        """
        Compute gradient of cov[i,j] respect to each element of lower_triang for each tasks i and j

        :param chol_base_cov_matrix: (np.array(n_tasks, n_tasks)). It's base_cov_matrix when
            the same_correlation is True.
        :param n_tasks: (int)
        :param same_correlation: (boolean)
        :return: {
            'entry (int)': np.array(number_tasks x number_tasks),
        }
        """

        gradient = {}

        if not same_correlation:
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


        gradient[0] = (chol_base_cov_matrix[0, 0]) * np.identity(n_tasks)

        if n_tasks == 1:
            return gradient

        gradient[1] = deepcopy(chol_base_cov_matrix)

        value = chol_base_cov_matrix[0, 1]

        gradient[0] = (chol_base_cov_matrix[0, 0] - value * (n_tasks - 1)) * np.identity(n_tasks)

        for i in xrange(n_tasks):
            gradient[1][i, i] = value * (n_tasks - 1)

        return gradient
