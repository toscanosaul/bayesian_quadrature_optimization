import unittest

import numpy.testing as npt


from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    TASKS_KERNEL_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    SAME_CORRELATION,
)
from stratified_bayesian_optimization.services.validate_gp_model import ValidateGPService


class TestValidateGPService(unittest.TestCase):

    def test_validate_gp_model(self):
        type_kernel = [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME]
        n_training = 3
        problem_name = "movies_collaborative"
        bounds_domain = [[0.01, 1.01], [0.1, 2.1], [1, 21], [1, 201], [0, 1, 2, 3, 4]]
        type_bounds = [0, 0, 0, 0, 1]
        dimensions = [5, 4, 5]
        thinning = 5
        n_burning = 100
        max_steps_out = 1000
        random_seed = 5
        training_name = None
        points = None
        noise = False
        n_samples = 0
        cache = True
        kernel_params = {SAME_CORRELATION: True}

        result = ValidateGPService.validate_gp_model(
            type_kernel, n_training, problem_name, bounds_domain, type_bounds, dimensions, thinning,
            n_burning, max_steps_out, random_seed, training_name, points, noise, n_samples, cache,
            **kernel_params)

        npt.assert_almost_equal(result, 0.666666666667)
