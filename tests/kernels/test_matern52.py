from __future__ import absolute_import

import unittest

from doubles import expect

import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.kernels.matern52 import Matern52, GradientLSMatern52
from stratified_bayesian_optimization.entities.parameter import ParameterEntity
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences


class TestMatern52(unittest.TestCase):

    def setUp(self):
        self.dimension = 2
        self.length_scale = ParameterEntity('scale', np.array([1.0, 2.0]), None)
        self.sigma2 = ParameterEntity('sigma2', np.array([3]), None)
        self.matern52 = Matern52(self.dimension, self.length_scale, self.sigma2)

        self.inputs = np.array([[1, 0], [0, 1]])

        self.matern52_ = Matern52(2, ParameterEntity('scale', np.array([2.0, 3.0]), None),
                                  ParameterEntity('sigma2', np.array([4.0]), None))

    def test_hypers(self):
        assert {'scale': self.length_scale, 'sigma2': self.sigma2} == self.matern52.hypers

    def test_set_parameters(self):
        length = ParameterEntity('scale_', np.array([1, 2]), None)
        sigma2 = ParameterEntity('sigma2', np.array([3]), None)

        self.matern52.set_parameters(length, sigma2)

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

        matern52 = Matern52(2, ParameterEntity('scale', np.array([2.0, 3.0]), None),
                            ParameterEntity('sigma2', np.array([4.0]), None))

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
            lambda params: Matern52.evaluate_cov_defined_by_params(params, inputs_1, 2),
            np.array([2.0, 3.0, 4.0]), np.array([dh]))

        gradient = Matern52.evaluate_grad_defined_by_params_respect_params(
            np.array([2.0, 3.0, 4.0]), inputs_1, 2)

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
            self.inputs, self.length_scale, self.sigma2) == {'scale': {0: 12, 1: 8}}

    def test_gradient_respect_distance(self):
        expect(GradientLSMatern52).gradient_respect_distance_cross.once().and_return(0)

        assert GradientLSMatern52.gradient_respect_distance(
            self.length_scale, self.sigma2, self.inputs) == 0

    def test_gradient_respect_distance_cross(self):
        expect(Distances).dist_square_length_scale.once().and_return(np.array([0.0]))

        assert GradientLSMatern52.gradient_respect_distance_cross(
            self.length_scale, self.sigma2, self.inputs, self.inputs) == np.array([0.0])

    def test_grad_respect_point_2(self):
        expect(GradientLSMatern52).gradient_respect_distance_cross.once().and_return(
            np.array([[1, 0], [0, 1]]))
        expect(Distances).gradient_distance_length_scale_respect_point.once().and_return(
            1.0
        )
        comparisons = GradientLSMatern52.grad_respect_point(self.length_scale, self.sigma2,
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
        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3, 5]))
        assert np.all(kernel.length_scale.value == np.array([1, 3]))
        assert kernel.sigma2.value == np.array([5])

    def test_evaluate_cov_defined_by_params(self):
        result = Matern52.evaluate_cov_defined_by_params(np.array([1, 3, 5]),
                                                         np.array([[4, 5]]), 2)

        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3, 5]))
        assert result == kernel.cov(np.array([[4, 5]]))

    def test_evaluate_grad_defined_by_params_respect_params(self):
        result = Matern52.evaluate_grad_defined_by_params_respect_params(
            np.array([1, 3, 5]), np.array([[4, 5]]), 2)
        kernel = Matern52.define_kernel_from_array(2, np.array([1, 3, 5]))

        grad_kernel = kernel.gradient_respect_parameters(np.array([[4, 5]]))
        assert result == {0: grad_kernel['length_scale'][0], 1: grad_kernel['length_scale'][1],
                          2: grad_kernel['sigma2']}
