from __future__ import absolute_import

import unittest

import numpy as np

from stratified_bayesian_optimization.lib.util import \
    convert_dictionary_gradient_to_simple_dictionary


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.dictionary = {'a': np.array([2]), 'b': {0: np.array([3]), 1: np.array([8])}}
        self.order_keys = [('b', [(0, None), (1, None)]), ('a', None)]

    def test_convert_dictionary_to_np_array(self):
        result = convert_dictionary_gradient_to_simple_dictionary(self.dictionary, self.order_keys)

        assert result == {0: np.array([3]), 1: np.array([8]), 2: np.array([2])}

