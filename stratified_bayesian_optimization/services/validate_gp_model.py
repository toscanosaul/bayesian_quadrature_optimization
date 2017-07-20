from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import (
    DEFAULT_RANDOM_SEED,
)
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.models.gp_fitting_gaussian import ValidationGPModel

logger = SBOLog(__name__)


class ValidateGPService(object):

    @classmethod
    def validate_gp_model(cls, type_kernel, n_training, problem_name, bounds_domain, type_bounds,
                          dimensions, thinning=0, n_burning=0, max_steps_out=1,
                          random_seed=None, training_name=None, points=None, noise=False,
                          n_samples=0, cache=True, **kernel_parameters):
        """

        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param n_training: int
        :param problem_name: str
        :param bounds_domain: [([float, float] or [float])], the first case is when the bounds are
            lower or upper bound of the respective entry; in the second case, it's list of finite
            points representing the domain of that entry.
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param thinning: (int)
        :param n_burning: (int) Number of burnings samples for the MCMC.
        :param max_steps_out: (int)  Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param random_seed: (int)
        :param training_name: str
        :param points: [[float]]. If training_data is None, we can evaluate the objective
            function in these points.
        :param noise: boolean
        :param n_samples: (int) If the objective is noisy, we take n_samples of the function to
            estimate its value.
        :param cache: (boolean)  Try to get trainng_data from cache if it's True
        :param kernel_parameters: additional kernel parameters,
            - SAME_CORRELATION: (boolean) True or False. Parameter used only for task kernel.

        :return: (int) percentage of success
        """

        if random_seed is None:
            random_seed = DEFAULT_RANDOM_SEED

        if training_name is None:
            training_name = 'default_training_data_%d_points_rs_%d' % (n_training, random_seed)

        training_data = TrainingDataService.get_training_data(problem_name, training_name,
                                                              bounds_domain,
                                                              n_training=n_training,
                                                              points=points,
                                                              noise=noise,
                                                              n_samples=n_samples,
                                                              random_seed=random_seed,
                                                              type_bounds=type_bounds,
                                                              cache=cache)

        training_data['evaluations'] = np.array(training_data['evaluations'])
        training_data['points'] = np.array(training_data['points'])

        if len(training_data['var_noise']) > 0:
            training_data['var_noise'] = np.array(training_data['var_noise'])
        else:
            training_data['var_noise'] = None

        results = ValidationGPModel.cross_validation_mle_parameters(
            type_kernel, training_data, dimensions, problem_name, bounds_domain, thinning,
            n_burning, max_steps_out, start=None, random_seed=random_seed,
            training_name=training_name, **kernel_parameters
        )

        logger.info('Percentage of success is: %f' % results['success_proportion'])

        return results['success_proportion']
