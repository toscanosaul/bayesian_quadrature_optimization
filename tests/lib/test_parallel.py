from __future__ import absolute_import

import unittest

from stratified_bayesian_optimization.lib.parallel import Parallel

def f(x):
    return x

def g(x):
    return x[1]

class TestParallel(unittest.TestCase):

    def test_run_function_different_arguments_parallel(self):
        arguments = {0: 1, 1: 2, 2: 3, 3: 4}

        result = Parallel.run_function_different_arguments_parallel(f, arguments)

        assert result == {0: 1, 1: 2, 2: 3, 3: 4}

        with self.assertRaises(Exception):
            Parallel.run_function_different_arguments_parallel(g, arguments, all_success=True)

        Parallel.run_function_different_arguments_parallel(g, arguments)
