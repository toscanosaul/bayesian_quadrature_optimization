import unittest

import numpy as np
import numpy.testing as npt

from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    SMALLEST_POSITIVE_NUMBER,
    CHOL_COV,
    SOL_CHOL_Y_UNBIASED,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
)
from stratified_bayesian_optimization.lib.finite_differences import FiniteDifferences


class TestGPFittingGaussian(unittest.TestCase):

    def setUp(self):
        type_kernel = [MATERN52_NAME]
        self.training_data = {
            "evaluations":[42.2851784656,72.3121248508,1.0113231069,30.9309246906,15.5288331909],
            "points":[
                [42.2851784656],[72.3121248508],[1.0113231069],[30.9309246906],[15.5288331909]],
            "var_noise":[]}
        dimensions = [1]

        self.gp = GPFittingGaussian(type_kernel, self.training_data, dimensions)

        self.training_data_3 = {
            "evaluations":[42.2851784656,72.3121248508,1.0113231069,30.9309246906,15.5288331909],
            "points":[
                [42.2851784656],[72.3121248508],[1.0113231069],[30.9309246906],[15.5288331909]],
            "var_noise":[0.5, 0.8, 0.7, 0.9, 1.0]}

        self.gp_3 = GPFittingGaussian(type_kernel, self.training_data_3, dimensions)
        self.training_data_simple = {
            "evaluations":[5],
            "points":[[5]],
            "var_noise":[]}
        dimensions = [1]

        self.simple_gp = GPFittingGaussian(type_kernel, self.training_data_simple, dimensions)

        self.training_data_complex = {
            "evaluations":[1.0],
            "points":[[42.2851784656, 0]],
            "var_noise":[0.5]}

        self.complex_gp = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            self.training_data_complex, [2, 1, 1])

        self.training_data_complex_2 = {
            "evaluations":[1.0, 2.0, 3.0],
            "points":[[42.2851784656, 0], [10.532, 0], [9.123123, 1]],
            "var_noise":[0.5, 0.2, 0.1]}

        self.complex_gp_2 = GPFittingGaussian(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            self.training_data_complex_2, [3, 1, 2])

        self.new_point = np.array([[80.0]])
        self.evaluation = np.array([80.0])

        self.training_data_noisy = {
            "evaluations":[41.0101845096],
            "points":[[42.2851784656]],
            "var_noise":[0.0181073779]}

        self.gp_noisy = GPFittingGaussian(type_kernel, self.training_data_noisy, dimensions)

        np.random.seed(2)
        normal_noise = np.random.normal(0,1.0,50)
        points_ = list(np.linspace(0, 10, 50))
        points = [[z] for z in points_]
        evaluations = list([point for point in points_] + normal_noise)

        self.gaussian_training_data = {
            "evaluations": evaluations,
            "points": points,
            "var_noise": []
        }

        self.gp_gaussian = GPFittingGaussian(type_kernel, self.gaussian_training_data, dimensions)


    def test_add_points_evaluations(self):

        self.gp.add_points_evaluations(self.new_point, self.evaluation)
        assert np.all(self.gp.data['evaluations'] == np.concatenate(
            (self.training_data['evaluations'], [80.0])))
        assert np.all(self.gp.data['points'] == np.concatenate((self.training_data['points'],
                                                              [[80.0]])))
        assert self.gp.data['var_noise'] is None

        assert self.gp.training_data == self.training_data

        self.gp_noisy.add_points_evaluations(self.new_point, self.evaluation, np.array([0.00001]))

        assert np.all(self.gp_noisy.data['evaluations'] == np.concatenate(
            (self.training_data_noisy['evaluations'], [80.0])))
        assert np.all(self.gp_noisy.data['points'] == np.concatenate(
            (self.training_data_noisy['points'], [[80.0]])))
        assert np.all(self.gp_noisy.data['var_noise'] == np.concatenate(
            (self.training_data_noisy['var_noise'], [0.00001])))

        assert self.gp_noisy.training_data == self.training_data_noisy

    def test_convert_from_list_to_numpy(self):
        data = GPFittingGaussian.convert_from_list_to_numpy(self.training_data_noisy)
        assert np.all(data['points'] == np.array([[42.2851784656]]))
        assert data['evaluations'] == np.array([41.0101845096])
        assert data['var_noise'] == np.array([0.0181073779])

        data_ = GPFittingGaussian.convert_from_list_to_numpy(self.training_data)
        assert np.all(data_['points'] == np.array([
                [42.2851784656],[72.3121248508],[1.0113231069],[30.9309246906],[15.5288331909]]))
        assert np.all(data_['evaluations'] == np.array([42.2851784656,72.3121248508,1.0113231069,
                                                        30.9309246906,15.5288331909]))
        assert data_['var_noise'] is None

    def test_convert_from_numpy_to_list(self):
        data = GPFittingGaussian.convert_from_list_to_numpy(self.training_data_noisy)
        data_list = GPFittingGaussian.convert_from_numpy_to_list(data)
        assert data_list == self.training_data_noisy

        data_ = GPFittingGaussian.convert_from_list_to_numpy(self.training_data)
        data_list_ = GPFittingGaussian.convert_from_numpy_to_list(data_)
        assert data_list_ == self.training_data

    def test_serialize(self):
        self.gp.add_points_evaluations(self.new_point, self.evaluation)
        dict = self.gp.serialize()
        assert dict == {
            'type_kernel': [MATERN52_NAME],
            'training_data': self.training_data,
            'dimensions': [1],
            'kernel_values': [1, 1],
            'mean_value': [0],
            'var_noise_value': [SMALLEST_POSITIVE_NUMBER],
            'thinning': 0,
            'data': {
            "evaluations":[42.2851784656,72.3121248508,1.0113231069,30.9309246906,15.5288331909,
                           80.0],
            "points":[
                [42.2851784656],[72.3121248508],[1.0113231069],[30.9309246906],[15.5288331909],
                [80.0]],
            "var_noise":[]},
        }

    def test_deserialize(self):
        params = {
            'type_kernel': [MATERN52_NAME],
            'training_data': self.training_data,
            'dimensions': [1],
        }
        gp = GPFittingGaussian.deserialize(params)

        assert gp.type_kernel == [MATERN52_NAME]
        assert gp.training_data == self.training_data
        assert gp.dimensions == [1]

    def test_get_parameters_model(self):
        parameters = self.gp.get_parameters_model
        parameters_values = [parameter.value for parameter in parameters]
        assert parameters_values[0] == np.array([SMALLEST_POSITIVE_NUMBER])
        assert parameters_values[1] == np.array([0.0])
        assert np.all(parameters_values[2] == np.array([1, 1]))

    def test_get_value_parameters_model(self):
        parameters = self.gp.get_value_parameters_model
        assert np.all(parameters == np.array([SMALLEST_POSITIVE_NUMBER, 0.0, 1, 1]))

    def test_cached_data(self):
        self.gp._updated_cached_data((3, 5, 1), -1, SOL_CHOL_Y_UNBIASED)
        assert self.gp.cache_sol_chol_y_unbiased[(3, 5, 1)] == -1
        assert self.gp.cache_sol_chol_y_unbiased.keys() == [(3, 5, 1)]
        assert self.gp.cache_chol_cov == {}
        assert self.gp._get_cached_data((3, 5, 1), SOL_CHOL_Y_UNBIASED) == -1

        self.gp._updated_cached_data((3, 5), 0, CHOL_COV)
        assert self.gp.cache_chol_cov[(3, 5)] == 0
        assert self.gp.cache_chol_cov.keys() == [(3, 5)]
        assert self.gp.cache_sol_chol_y_unbiased == {}
        assert self.gp._get_cached_data((3, 5), CHOL_COV) == 0

        assert self.gp._get_cached_data((3, 0), CHOL_COV) is False

    def test_chol_cov_including_noise(self):
        chol, cov = self.simple_gp._chol_cov_including_noise(1.0, np.array([1.0, 1.0]))
        assert cov == np.array([[2.0]])
        assert chol == np.array([[np.sqrt(2.0)]])

        chol, cov = self.simple_gp._chol_cov_including_noise(1.0, np.array([1.0, 1.0]))
        assert cov == np.array([[2.0]])
        assert chol == np.array([[np.sqrt(2.0)]])

        chol, cov = self.complex_gp._chol_cov_including_noise(1.0, np.array([1.0, 1.0, 0.0]))
        assert cov == np.array([[2.5]])
        assert chol == np.array([[np.sqrt(2.5)]])

    def test_log_likelihood(self):
        llh = self.complex_gp.log_likelihood(1.0, 1.0, np.array([1.0, 1.0, 0.0]))
        assert llh == -0.45814536593707761

        llh = self.complex_gp.log_likelihood(1.0, 1.0, np.array([1.0, 1.0, 0.0]))
        assert llh == -0.45814536593707761

    def test_grad_log_likelihood(self):
        grad = self.complex_gp_2.grad_log_likelihood(1.0, 1.0, np.array([1.0, 1.0, 0.0, 0.0, 0.0]))

        dh = 0.0000001
        finite_diff = FiniteDifferences.forward_difference(
            lambda params: self.complex_gp_2.log_likelihood(
                params[0], params[1], params[2:]
            ),
            np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), np.array([dh]))

        for i in range(7):
            npt.assert_almost_equal(finite_diff[i], grad[i])

        grad_2 = self.complex_gp_2.grad_log_likelihood(1.82, 123.1,
                                                       np.array([5.0, 7.3, 1.0, -5.5, 10.0]))

        dh = 0.00000001
        finite_diff_2 = FiniteDifferences.forward_difference(
            lambda params: self.complex_gp_2.log_likelihood(
                params[0], params[1], params[2:]
            ),
            np.array([1.82, 123.1, 5.0, 7.3, 1.0, -5.5, 10.0]), np.array([dh]))

        for i in range(7):
            npt.assert_almost_equal(finite_diff_2[i], grad_2[i], decimal=5)

        grad_3 = self.gp_3.grad_log_likelihood(1.82, 123.1, np.array([5.0, 7.3]))
        dh = 0.0000001
        finite_diff_3 = FiniteDifferences.forward_difference(
            lambda params: self.gp_3.log_likelihood(
                params[0], params[1], params[2:]
            ),
            np.array([1.82, 123.1, 5.0, 7.3]), np.array([dh]))
        for i in range(4):
            npt.assert_almost_equal(finite_diff_3[i], grad_3[i], decimal=5)

    def test_grad_log_likelihood_dict(self):
        grad = self.complex_gp_2.grad_log_likelihood_dict(1.82, 123.1,
                                                       np.array([5.0, 7.3, 1.0, -5.5, 10.0]))
        grad_2 = self.complex_gp_2.grad_log_likelihood(1.82, 123.1,
                                                       np.array([5.0, 7.3, 1.0, -5.5, 10.0]))

        assert grad_2[0] == grad['var_noise']
        assert grad_2[1] == grad['mean']
        assert np.all(grad_2[2:] == grad['kernel_params'])

    def test_mle_parameters(self):
        print self.gp_gaussian.log_likelihood(1.0, 0.0, np.array([1.0, 3.0]))
        opt = self.gp_gaussian.mle_parameters(start=np.array([1.0, 0.0, 1.0, 3.0]))
        #opt = self.gp_gaussian.mle_parameters()
        print opt


