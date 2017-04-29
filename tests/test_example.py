import unittest

import stratified_bayesian_optimization.example


class TestExample(unittest.TestCase):

    def test(self):
        assert stratified_bayesian_optimization.example.fun() == 27
