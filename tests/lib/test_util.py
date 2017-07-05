from __future__ import absolute_import

import unittest
from mock import create_autospec
from doubles import expect

from stratified_bayesian_optimization.lib.util import *
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.lib.constant import (
    PRODUCT_KERNELS_SEPARABLE,
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
)


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.dictionary = {'a': np.array([2]), 'b': {0: np.array([3]), 1: np.array([8])}}
        self.order_keys = [('b', [(0, None), (1, None)]), ('a', None)]
        self.gp_regression = create_autospec(GPFittingGaussian)

    def test_convert_dictionary_to_np_array(self):
        result = convert_dictionary_gradient_to_simple_dictionary(self.dictionary, self.order_keys)

        assert result == {0: np.array([3]), 1: np.array([8]), 2: np.array([2])}

    def test_separate_numpy_arrays_in_lists(self):
        separate = separate_numpy_arrays_in_lists(np.array([1,2,3]),2)
        assert np.all(separate[0] == np.array([1, 2]))
        assert np.all(separate[1] == np.array([3]))
        assert len(separate) == 2

        separate = separate_numpy_arrays_in_lists(np.array([[1,2,3], [4, 5, 6]]),2)
        assert np.all(separate[0] == np.array([[1, 2], [4,5]]))
        assert np.all(separate[1] == np.array([[3], [6]]))
        assert len(separate) == 2

    def test_wrapper_fit_gp_regression(self):
        expect(self.gp_regression).fit_gp_regression.once().and_return(self.gp_regression)
        assert wrapper_fit_gp_regression(self.gp_regression) == self.gp_regression

    def test_wrapper_evaluate_objective_function(self):
        expect(TrainingDataService).evaluate_function.once().and_return(0)
        assert wrapper_evaluate_objective_function(0, TrainingDataService,
                                                   "problems.test_problem.main", 0) == 0

    def test_get_number_parameters_kernel(self):
        assert get_number_parameters_kernel(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME], [2, 1, 1]) == 3

        with self.assertRaises(NameError):
            get_number_parameters_kernel(['a'], [2])

    def test_get_default_values_kernel(self):
        assert get_default_values_kernel([MATERN52_NAME], [1]) == [1, 1]
        assert get_default_values_kernel([TASKS_KERNEL_NAME], [1]) == [0]
        assert get_default_values_kernel(
            [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME], [2, 1, 1]) == [1, 1, 0]

    def test_convert_list_to_dictionary(self):
        assert convert_list_to_dictionary(['a', 'b']) == {0: 'a', 1: 'b'}

    def test_convert_dictionary_to_list(self):
        assert convert_dictionary_to_list({0: 'a', 1: 'b'}) == ['a', 'b']

    def test_expand_dimension_vector(self):
        x = np.array([1, 2, 3])
        default_x = np.array([1, 9, 8, 7, 10, 11])
        change_indexes = [1, 3, 5]

        assert np.all(expand_dimension_vector(x, change_indexes, default_x) == \
               np.array([1, 1, 8, 2, 10, 3]))

    def test_reduce_dimension_vector(self):
        x = np.array([1, 9, 8, 7, 10, 11])
        change_indexes = [1, 3, 5]

        assert np.all(reduce_dimension_vector(x, change_indexes) == np.array([9, 7, 11]))

    def test_combine_vectors(self):
        a = np.array([1, 5])
        b = np.array([2])
        indexes = [0, 2]

        assert np.all(combine_vectors(a, b, indexes) == np.array([1, 2, 5]))

    def test_separate_vector(self):
        a = np.array([1, 2, 5])
        indexes = [0, 2]

        a1 = np.array([1, 5])
        a2 = np.array([2])

        result = separate_vector(a, indexes)

        assert np.all(result[0] == a1)
        assert np.all(result[1] == a2)
        assert len(result) == 2
