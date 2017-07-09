import unittest

from mock import create_autospec
from doubles import expect

import numpy as np
import numpy.testing as npt
import copy

from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.kernels.product_kernels import ProductKernels
from stratified_bayesian_optimization.kernels.matern52 import Matern52
from stratified_bayesian_optimization.kernels.tasks_kernel import TasksKernel
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    LENGTH_SCALE_NAME,
    SIGMA2_NAME,
    LOWER_TRIANG_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
)
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences


class TestProductKernels(unittest.TestCase):

    def setUp(self):
        self.length_scale = ParameterEntity('scale', np.array([1.0]), None)
        self.sigma2 = ParameterEntity('sigma2', np.array([3]), None)
        self.matern52 = Matern52(1, self.length_scale)
        self.matern52 = ScaledKernel(1, self.matern52, self.sigma2)

        self.n_tasks = 1
        self.lower_triang = ParameterEntity('lower_triang', np.array([1.0]), None)
        self.task_kernel = TasksKernel(self.n_tasks, self.lower_triang)

        self.tasks = create_autospec(TasksKernel)
        self.kernel = ProductKernels(self.matern52, self.task_kernel)

        self.prior = UniformPrior(2, [1, 1], [100, 100])
        self.prior_2 = UniformPrior(1, [1], [100])
        self.prior_3 = UniformPrior(3, [1, 1, 1], [100, 100, 100])

        self.length_scale_ = ParameterEntity(LENGTH_SCALE_NAME, np.array([1.0, 5.0]), self.prior)
        self.sigma2_ = ParameterEntity(SIGMA2_NAME, np.array([3]), self.prior_2)
        self.matern52_ = Matern52(2, self.length_scale_)
        self.matern52_ = ScaledKernel(2, self.matern52_, self.sigma2_)

        self.n_tasks_ = 2
        self.lower_triang_ = ParameterEntity(LOWER_TRIANG_NAME, np.array([1.0, 5.0, 6.0]),
                                             self.prior_3)
        self.task_kernel_ = TasksKernel(self.n_tasks_, self.lower_triang_)

        self.kernel_ = ProductKernels(self.matern52_, self.task_kernel_)

        self.matern52_ver2_ = Matern52(2, self.length_scale_)

        self.kernel_ver2 = ProductKernels(self.matern52_ver2_, self.task_kernel_)

        self.parameters = {MATERN52_NAME: {
            'kernel_parameters': [ParameterEntity('scale', np.array([5.0]), None)],
            'sigma2': self.sigma2},
            TASKS_KERNEL_NAME: {
                'lower_triang': ParameterEntity('lower_triang', np.array([10.0]), None)
            }
        }

        self.inputs = {MATERN52_NAME: np.array([[5.0]]), TASKS_KERNEL_NAME: np.array([[0]])}

        self.inputs_ = {MATERN52_NAME: np.array([[5.0, 6.0], [8.0, 9.0]]),
                        TASKS_KERNEL_NAME: np.array([[0], [1]])}

    def test_hypers(self):
        assert self.kernel.hypers == {MATERN52_NAME: {
            'scale': self.length_scale, 'sigma2': self.sigma2},
            TASKS_KERNEL_NAME: {
                'lower_triang': self.lower_triang
            }
        }

    def test_name_parameters_as_list(self):
        assert self.kernel_.name_parameters_as_list == \
               [(LENGTH_SCALE_NAME, [(0, None), (1, None)]), (SIGMA2_NAME, None)] + \
               [(LOWER_TRIANG_NAME, [(i, None) for i in xrange(3)])]

    def test_set_parameters(self):
        self.kernel.set_parameters(self.parameters)

        assert self.kernel.hypers == {MATERN52_NAME: {
            'scale': self.parameters[MATERN52_NAME]['kernel_parameters'][0], 'sigma2': self.sigma2},
            TASKS_KERNEL_NAME: {
                'lower_triang': self.parameters[TASKS_KERNEL_NAME]['lower_triang']
            }
        }

    def test_define_kernel_from_array(self):
        kernel = ProductKernels.define_kernel_from_array(
            [2, 2], [np.array([1.0, 5.0, 3.0]), np.array([1.0, 5.0, 6.0])],
            [MATERN52_NAME, TASKS_KERNEL_NAME])

        assert kernel.name == self.kernel_.name
        assert kernel.dimension == self.kernel_.dimension
        assert kernel.names == self.kernel_.names

        comparisons = self.kernel_.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value == \
            kernel.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value
        assert np.all(comparisons)

        comparisons = self.kernel_.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value == \
            kernel.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value
        assert np.all(comparisons)

    def test_cov_dict(self):
        expect(self.kernel).cross_cov_dict.once().and_return(0)
        assert self.kernel.cov_dict(self.inputs) == 0

    def test_cross_cov_dict(self):
        expect(self.matern52).cross_cov.once().and_return(5)
        expect(self.task_kernel).cross_cov.once().and_return(10)

        assert self.kernel.cross_cov_dict(self.inputs, self.inputs) == 50

    def test_gradient_respect_parameters(self):
        expect(self.matern52).cov.once().and_return(5)
        expect(self.task_kernel).cov.once().and_return(10)

        expect(self.matern52).gradient_respect_parameters.once().and_return({
            'scale': 2.0,
            'sigma2': 3.0
        })
        expect(self.task_kernel).gradient_respect_parameters.once().and_return({
            'lower_triang': -1.0
        })

        assert self.kernel.gradient_respect_parameters(self.inputs) == {MATERN52_NAME: {
            'scale': 20.0, 'sigma2': 30.0
        }, TASKS_KERNEL_NAME: {'lower_triang': -5.0}}

    def test_grad_respect_point_dict(self):
        expect(self.matern52).cross_cov.once().and_return(5)
        expect(self.task_kernel).cross_cov.once().and_return(10)

        expect(self.matern52).grad_respect_point.once().and_return(-1)
        expect(self.task_kernel).grad_respect_point.once().and_return(3)

        assert self.kernel.grad_respect_point_dict(self.inputs, self.inputs) == \
            {MATERN52_NAME: -10, TASKS_KERNEL_NAME: 15}

    def test_evaluate_cov_defined_by_params(self):
        result = ProductKernels.evaluate_cov_defined_by_params(
            [np.array([1.0, 5.0]), np.array([1.0, 5.0, 6.0])],
            self.inputs_, [2, 2], [MATERN52_NAME, TASKS_KERNEL_NAME])

        assert np.all(result == self.kernel_ver2.cov_dict(self.inputs_))

    def test_evaluate_grad_defined_by_params_respect_params(self):
        result = ProductKernels.evaluate_grad_defined_by_params_respect_params(
            [np.array([1.0, 5.0]), np.array([1.0, 5.0, 6.0])],
            self.inputs_, [2, 2], [MATERN52_NAME, TASKS_KERNEL_NAME])

        grad_kernel = self.kernel_ver2.gradient_respect_parameters(self.inputs_)

        for i in range(2):
            assert np.all(result[i] == grad_kernel[MATERN52_NAME][LENGTH_SCALE_NAME][i])

        for i in range(3):
            assert np.all(result[i + 2] == grad_kernel[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME][i])

    def test_gradient_respect_parameters_finite_differences(self):
        inputs_1 = self.inputs_
        dh = np.array(5 * [0.00000001])
        dh[2] = 0.0001
        dh[4] = 0.00000001

        params_ = [np.array([1.0, 5.0]), np.array([1.0, 5.0, 6.0])]

        gradient = ProductKernels.evaluate_grad_defined_by_params_respect_params(
            params_, self.inputs_, [2, 2], [MATERN52_NAME, TASKS_KERNEL_NAME]
        )

        finite_diff = FiniteDifferences.forward_difference(
            lambda params: ProductKernels.evaluate_cov_defined_by_params(
                [params[0: 2], params[2: 5]], inputs_1, [2, 2],
                [MATERN52_NAME, TASKS_KERNEL_NAME]),
            np.array([1.0, 5.0, 1.0, 5.0, 6.0]), np.array([dh]))

        for i in range(2):
            npt.assert_almost_equal(finite_diff[i], gradient[i], decimal=5)

        npt.assert_almost_equal(finite_diff[2], gradient[2], decimal=2)

        npt.assert_almost_equal(finite_diff[3], gradient[3], decimal=3)
        npt.assert_almost_equal(finite_diff[4], gradient[4], decimal=2)

    def test_cov(self):
        expect(self.kernel).cross_cov.once().and_return(0)
        assert self.kernel.cov(self.inputs) == 0

    def test_cross_cov(self):
        inputs = np.array([[5.0, 0]])
        assert self.kernel.cross_cov(inputs, inputs) == \
            self.kernel.cross_cov_dict(self.inputs, self.inputs)

    def test_grad_respect_point(self):
        dh = [0.000000000001, 0.000000000001, 0.0000001]
        inputs_1 = np.array([[2.0, 4.0, 0], [3.0, 5.0, 1]])
        point_ = np.array([[42.0, 35.0, 1]])

        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.kernel_.cross_cov(point.reshape([1, 3]), inputs_1),
            np.array([42.0, 35.0, 1]), np.array([dh]))

        gradient = self.kernel_.grad_respect_point(point_, inputs_1)

        for i in range(2):
            npt.assert_almost_equal(finite_diff[i], gradient[:, i:i+1].transpose())

    def test_hypers_as_list(self):
        assert self.kernel_.hypers_as_list == [self.length_scale_, self.sigma2_, self.lower_triang_]

    def test_hypers_values_as_array(self):
        assert np.all(self.kernel_.hypers_values_as_array == np.array([1, 5, 3, 1, 5, 6]))

    def test_sample_parameters(self):
        parameters = [self.length_scale_,  self.sigma2_, self.lower_triang_]
        samples = []
        np.random.seed(1)
        for parameter in parameters:
            samples.append(parameter.sample_from_prior(2))

        assert np.all(self.kernel_.sample_parameters(2, random_seed=1) == np.array([
            [samples[0][0, 0], samples[0][0, 1], samples[1][0], samples[2][0, 0], samples[2][0, 1],
             samples[2][0, 2]],
            [samples[0][1, 0], samples[0][1, 1], samples[1][1], samples[2][1, 0], samples[2][1, 1],
             samples[2][1, 2]]]))

    def test_get_bounds_parameters(self):
        assert self.kernel_.get_bounds_parameters() == 6 * [(SMALLEST_NUMBER, LARGEST_NUMBER)]

    def test_define_default_kernel(self):
        kern1 = ProductKernels.define_default_kernel([1, 1], None, None, None,
                                                     [MATERN52_NAME, TASKS_KERNEL_NAME])

        assert kern1.name == PRODUCT_KERNELS_SEPARABLE + ':' + MATERN52_NAME + '_' + \
                             TASKS_KERNEL_NAME + '_'
        assert kern1.dimension == 2
        assert kern1.dimension_parameters == 2
        assert kern1.names == [MATERN52_NAME, TASKS_KERNEL_NAME]

        assert kern1.kernels[MATERN52_NAME].length_scale.value == np.array([1])
        assert kern1.kernels[TASKS_KERNEL_NAME].lower_triang.value == np.array([0])

        assert kern1.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value == np.array([1])
        assert kern1.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value == np.array([0])

        kern2 = ProductKernels.define_default_kernel([1, 1], None,
                                                     [np.array([3]), np.array([5])], None,
                                                     [MATERN52_NAME, TASKS_KERNEL_NAME])

        assert kern2.name == PRODUCT_KERNELS_SEPARABLE + ':' + MATERN52_NAME + '_' + \
                             TASKS_KERNEL_NAME + '_'
        assert kern2.dimension == 2
        assert kern2.dimension_parameters == 2
        assert kern2.names == [MATERN52_NAME, TASKS_KERNEL_NAME]

        assert kern2.kernels[MATERN52_NAME].length_scale.value == np.array([3])
        assert kern2.kernels[TASKS_KERNEL_NAME].lower_triang.value == np.array([5])

        assert kern2.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value == np.array([3])
        assert kern2.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value == np.array([5])

    def test_update_value_parameters(self):
        self.kernel_.update_value_parameters(np.array([5, 8, 7, 15, 16, 17]))

        assert np.all(self.kernel_.kernels[MATERN52_NAME].kernel.length_scale.value ==
                      np.array([5, 8]))
        assert self.kernel_.kernels[MATERN52_NAME].sigma2.value == np.array([7])
        assert np.all(self.kernel_.kernels[TASKS_KERNEL_NAME].lower_triang.value
                      == np.array([15, 16, 17]))

        assert np.all(self.kernel_.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value
                      == np.array([5, 8]))
        assert self.kernel_.parameters[MATERN52_NAME][SIGMA2_NAME].value == np.array([7])
        assert np.all(self.kernel_.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value ==
                      np.array([15, 16, 17]))

    def test_compare_kernels(self):
        kernel = ProductKernels.define_kernel_from_array([1, 1], [np.array([1]), np.array([0])],
                                                         [MATERN52_NAME, TASKS_KERNEL_NAME])

        kernel_ = copy.deepcopy(kernel)
        kernel_.name = 'a'
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension = 3
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension_parameters = 5
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.names = ['a', 'b']
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.kernels[MATERN52_NAME].name = 'a'
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.kernels[TASKS_KERNEL_NAME].name = 'a'
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.parameters[MATERN52_NAME][LENGTH_SCALE_NAME].value = [2]
        assert ProductKernels.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.parameters[TASKS_KERNEL_NAME][LOWER_TRIANG_NAME].value = [3]
        assert ProductKernels.compare_kernels(kernel, kernel_) is False


