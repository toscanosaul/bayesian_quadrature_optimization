from __future__ import absolute_import

import unittest

from doubles import expect

import copy
import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.scaled_kernel import ScaledKernel
from stratified_bayesian_optimization.kernels.matern52 import Matern52, GradientLSMatern52
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences
from stratified_bayesian_optimization.priors.uniform import UniformPrior
from stratified_bayesian_optimization.lib.constant import (
    SMALLEST_NUMBER,
    LARGEST_NUMBER,
    MATERN52_NAME,
    SMALLEST_POSITIVE_NUMBER,
    LENGTH_SCALE_NAME,
)


class TestMatern52(unittest.TestCase):

    def setUp(self):
        self.dimension = 2
        self.length_scale = ParameterEntity('scale', np.array([1.0, 2.0]), None)
        self.sigma2 = ParameterEntity('sigma2', np.array([3]), None)
        self.matern52 = Matern52(self.dimension, self.length_scale)
        self.matern52 = ScaledKernel(self.dimension, self.matern52, self.sigma2)

        self.inputs = np.array([[1, 0], [0, 1]])

        self.prior = UniformPrior(2, [1, 1], [100, 100])
        self.prior_2 = UniformPrior(1, [1], [100])
        self.matern52_ = Matern52(2, ParameterEntity(LENGTH_SCALE_NAME,
                                                     np.array([2.0, 3.0]), self.prior))
        self.matern52_ = ScaledKernel(
            self.dimension, self.matern52_,
            ParameterEntity('sigma2', np.array([4.0]), self.prior_2))

    def test_hypers(self):
        assert {'scale': self.length_scale, 'sigma2': self.sigma2} == self.matern52.hypers

    def test_set_parameters(self):
        length = ParameterEntity('scale_', np.array([1, 2]), None)
        sigma2 = ParameterEntity('sigma2', np.array([3]), None)

        self.matern52.set_parameters([length], sigma2=sigma2)

        assert self.matern52.hypers == {'scale_': length, 'sigma2': sigma2}

    def test_cov(self):
        expect(self.matern52).cross_cov.once().and_return(0)
        assert self.matern52.cov(self.inputs) == 0

    def test_cross_cov(self):
        r2 = np.array([[0.0, 1.25], [1.25, 0.0]])
        r = np.sqrt(r2)

        left_term = ((1.0 + np.sqrt(5)*r + (5.0/3.0)*r2) * np.exp(-np.sqrt(5) * r) *
                     np.array([3]))[0, 1]
        comparisons = left_term == self.matern52.cross_cov(self.inputs, self.inputs)[0, 1]
        assert np.all(comparisons)

        point_1 = np.array([[2.0, 4.0]])
        point_2 = np.array([[3.0, 5.0]])

        matern52 = Matern52(2, ParameterEntity('scale', np.array([2.0, 3.0]), None))
        matern52 = ScaledKernel(2, matern52, ParameterEntity('sigma2', np.array([4.0]), None))

        assert np.all(matern52.cross_cov(point_1, point_2) == np.array([[3.0737065834936015]]))

        inputs_1 = np.array([[2.0, 4.0], [3.0, 5.0]])
        inputs_2 = np.array([[1.5, 9.0], [-3.0, 8.0]])

        assert np.all(matern52.cross_cov(inputs_1, inputs_2) ==
                      np.array([[0.87752659905500319, 0.14684671522649542],
                                [1.0880320585678382, 0.084041575076539962]]))

        inputs_1 = np.array([[2.0, 4.0]])
        inputs_2 = np.array([[1.5, 9.0], [-3.0, 8.0]])

        assert np.all(matern52.cross_cov(inputs_1, inputs_2) ==
                      np.array([[0.87752659905500319, 0.14684671522649542]]))

        inputs_1 = np.array([[2.0, 4.0], [3.0, 5.0]])
        inputs_2 = np.array([[1.5, 9.0]])

        npt.assert_almost_equal(matern52.cross_cov(inputs_1, inputs_2),
                                np.array([[0.87752659905500319], [1.0880320585678382]]))

    def test_gradient_respect_parameters(self):
        expect(GradientLSMatern52).gradient_respect_parameters_ls.once().and_return({'a': 0})
        expect(self.matern52).cov.once().and_return(1.0)

        assert self.matern52.gradient_respect_parameters(self.inputs) == {'a': 0, 'sigma2': 1.0/3}

    def test_gradient_respect_parameters_finite_differences(self):
        inputs_1 = np.array([[2.0, 4.0], [3.0, 5.0]])
        dh = 0.00000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda params:
            ScaledKernel.evaluate_cov_defined_by_params(params, inputs_1, 2, *([MATERN52_NAME],)),
            np.array([2.0, 3.0, 4.0]), np.array([dh]))

        gradient = ScaledKernel.evaluate_grad_defined_by_params_respect_params(
            np.array([2.0, 3.0, 4.0]), inputs_1, 2, *([MATERN52_NAME],))

        for i in range(3):
            npt.assert_almost_equal(finite_diff[i], gradient[i])

    def test_grad_respect_point(self):
        expect(GradientLSMatern52).grad_respect_point.once().and_return(0)

        assert 0 == self.matern52.grad_respect_point(self.inputs, self.inputs)

    def test_grad_respect_point_finite_differences(self):
        dh = 0.000000000001
        inputs_1 = np.array([[2.0, 4.0], [3.0, 5.0]])
        point = np.array([[42.0, 35.0]])
        finite_diff = FiniteDifferences.forward_difference(
            lambda point: self.matern52_.cross_cov(point.reshape([1, 2]), inputs_1),
            np.array([42.0, 35.0]), np.array([dh]))

        gradient = self.matern52_.grad_respect_point(point, inputs_1)
        for i in range(2):
            npt.assert_almost_equal(finite_diff[i], gradient[:, i:i+1].transpose())

    def test_gradient_respect_parameters_ls(self):
        expect(GradientLSMatern52).gradient_respect_distance.once().and_return(4)
        expect(Distances).gradient_distance_length_scale_respect_ls.once().and_return(
            {0: 3, 1: 2}
        )

        assert GradientLSMatern52.gradient_respect_parameters_ls(
            self.inputs, self.length_scale) == {'scale': {0: 12, 1: 8}}

    def test_gradient_respect_distance(self):
        expect(GradientLSMatern52).gradient_respect_distance_cross.once().and_return(0)

        assert GradientLSMatern52.gradient_respect_distance(
            self.length_scale, self.inputs) == 0

    def test_gradient_respect_distance_cross(self):
        expect(Distances).dist_square_length_scale.once().and_return(np.array([0.0]))

        assert GradientLSMatern52.gradient_respect_distance_cross(
            self.length_scale, self.inputs, self.inputs) == np.array([0.0])

    def test_grad_respect_point_2(self):
        expect(GradientLSMatern52).gradient_respect_distance_cross.once().and_return(
            np.array([[1, 0], [0, 1]]))
        expect(Distances).gradient_distance_length_scale_respect_point.once().and_return(
            1.0
        )
        comparisons = GradientLSMatern52.grad_respect_point(self.length_scale,
                                                            self.inputs, self.inputs) == \
            np.array([[1, 0], [0, 1]])
        assert np.all(comparisons)

    def test_grad_respect_point_matern(self):
        expect(GradientLSMatern52).grad_respect_point.once().and_return(0.0)

        assert self.matern52.grad_respect_point(self.inputs, self.inputs) == 0.0

    def test_name_parameters_as_list(self):
        assert self.matern52.name_parameters_as_list == \
               [('scale', [(0, None), (1, None)]), ('sigma2', None)]

    def test_define_kernel_from_array(self):
        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3]))
        assert np.all(kernel.length_scale.value == np.array([1, 3]))

    def test_evaluate_cov_defined_by_params(self):
        result = Matern52.evaluate_cov_defined_by_params(np.array([1, 3, 5]),
                                                         np.array([[4, 5]]), 2)

        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3, 5]))
        assert result == kernel.cov(np.array([[4, 5]]))

    def test_evaluate_grad_defined_by_params_respect_params(self):
        result = Matern52.evaluate_grad_defined_by_params_respect_params(
            np.array([1, 3]), np.array([[4, 5]]), 2)
        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3]))

        grad_kernel = kernel.gradient_respect_parameters(np.array([[4, 5]]))
        assert result == {0: grad_kernel['length_scale'][0], 1: grad_kernel['length_scale'][1]}

    def test_hypers_as_list(self):

        assert self.matern52_.hypers_as_list == [self.matern52_.kernel.length_scale,
                                                 self.matern52_.sigma2]

    def test_hypers_values_as_array(self):
        assert np.all(self.matern52_.hypers_values_as_array == np.array([2.0, 3.0, 4.0]))

    def test_sample_parameters(self):
        parameters = self.matern52_.hypers_as_list
        samples = []
        np.random.seed(1)
        for parameter in parameters:
            samples.append(parameter.sample_from_prior(2))
        assert np.all(self.matern52_.sample_parameters(2, random_seed=1) == np.array([
            [samples[0][0, 0], samples[0][0, 1], samples[1][0]],
            [samples[0][1, 0], samples[0][1, 1], samples[1][1]]]))

        np.random.seed(1)
        matern52 = Matern52(2, ParameterEntity(LENGTH_SCALE_NAME, np.array([2.0, 3.0]), self.prior))
        samples = []
        parameters1 = matern52.hypers_as_list
        for parameter in parameters1:
            samples.append(parameter.sample_from_prior(2))
        assert np.all(matern52.sample_parameters(2, random_seed=1) == samples[0])

    def test_get_bounds_parameters(self):
        assert self.matern52_.get_bounds_parameters() == 3 * [(SMALLEST_NUMBER, LARGEST_NUMBER)]

    def test_update_value_parameters(self):
        self.matern52_.update_value_parameters(np.array([1, 5, 10]))

        assert self.matern52_.sigma2.value == np.array([10])
        parameters = self.matern52_.hypers
        assert np.all(parameters[LENGTH_SCALE_NAME].value == np.array([1, 5]))

    def test_define_default_kernel(self):
        kern1 = Matern52.define_default_kernel(1)

        assert kern1.name == MATERN52_NAME
        assert kern1.dimension == 1
        assert kern1.dimension_parameters == 1
        assert kern1.length_scale.value == np.array([1])
        assert kern1.length_scale.prior.max == [LARGEST_NUMBER]
        assert kern1.length_scale.prior.min == [SMALLEST_POSITIVE_NUMBER]

        kern2 = Matern52.define_default_kernel(1, default_values=np.array([5]))

        assert kern2.name == MATERN52_NAME
        assert kern2.dimension == 1
        assert kern2.dimension_parameters == 1
        assert kern2.length_scale.value == np.array([5])
        assert kern2.length_scale.prior.max == [LARGEST_NUMBER]
        assert kern2.length_scale.prior.min == [SMALLEST_POSITIVE_NUMBER]

        kern3 = Matern52.define_default_kernel(1, bounds=[[5, 6]])
        assert kern3.name == MATERN52_NAME
        assert kern3.dimension == 1
        assert kern3.dimension_parameters == 1
        assert kern3.length_scale.value == np.array([1])
        assert kern3.length_scale.prior.max == [20.0]
        assert kern3.length_scale.prior.min == [SMALLEST_POSITIVE_NUMBER]

    def test_compare_kernels(self):
        kernel = Matern52.define_kernel_from_array(1, np.ones(1))

        kernel_ = copy.deepcopy(kernel)
        kernel_.name = 'a'
        assert Matern52.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension = 2
        assert Matern52.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.dimension_parameters = 5
        assert Matern52.compare_kernels(kernel, kernel_) is False

        kernel_ = copy.deepcopy(kernel)
        kernel_.length_scale.value = np.array([-1])
        assert Matern52.compare_kernels(kernel, kernel_) is False

    def test_define_prior_parameters(self):
        data = {
            'points': np.array([[1]]),
            'evaluations': np.array([1]),
            'var_noise': None,
        }

        dimension = 1

        result = Matern52.define_prior_parameters(data, dimension)

        assert result == {
            LENGTH_SCALE_NAME: [0.0],
        }

        data2 = {
            'points': np.array([[1], [2]]),
            'evaluations': np.array([1, 2]),
            'var_noise': None,
        }

        dimension2 = 1

        result2 = Matern52.define_prior_parameters(data2, dimension2)

        assert result2 == {
            LENGTH_SCALE_NAME: [1.5432098765432098],
        }

    def test_evaluate_grad_respect_point(self):
        result = Matern52.evaluate_grad_respect_point(np.array([5.0]), np.array([[1]]),
                                                           np.array([[4], [5]]), 1)

        kernel = Matern52.define_kernel_from_array(1, np.array([5.0]))
        assert np.all(result == kernel.grad_respect_point(np.array([[1]]), np.array([[4], [5]])))

    def test_evaluate_hessian_respect_point(self):
        point = np.array([[4.5, 7.5]])
        inputs = np.array([[5.0, 6.0], [8.0, 9.0]])
        params = np.array([1.0, 5.0])
        result = Matern52.evaluate_hessian_respect_point(
            params, point, inputs, 2)


        dh = 0.00001
        finite_diff = FiniteDifferences.second_order_central(
            lambda x: Matern52.evaluate_cross_cov_defined_by_params(
                params, x.reshape((1, len(x))), inputs, 2),
            point[0, :], np.array([dh])
        )

        for i in xrange(2):
            for j in xrange(2):
                print i, j
                npt.assert_almost_equal(finite_diff[i, j],
                                        np.array([[result[0][i, j], result[1][i, j]]]), decimal=5)

    def test_hessian_distance_length_scale_respect_point(self):
        params = np.array([1.0, 5.0])
        point = np.array([[4.5, 7.5]])
        inputs = np.array([[5.0, 6.0], [8.0, 9.0]])
        result = Distances.gradient_distance_length_scale_respect_point(
            params, point, inputs, second=True
        )
        result = result['second']

        dh = 0.00001
        finite_diff = FiniteDifferences.second_order_central(
            lambda x: np.sqrt(Distances.dist_square_length_scale(
                params, x.reshape((1, len(x))), inputs)),
            point[0, :], np.array([dh])
        )

        for i in xrange(2):
            for j in xrange(2):
                print i, j
                npt.assert_almost_equal(finite_diff[i, j],
                                        np.array([[result[0][i, j], result[1][i, j]]]), decimal=5)
