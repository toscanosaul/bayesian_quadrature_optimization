from __future__ import absolute_import

from os import path

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import GP_DIR
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.lib.constant import DEFAULT_RANDOM_SEED

logger = SBOLog(__name__)


class GPFittingService(object):
    _filename = 'gp_{model_type}_{problem_name}_{type_kernel}_{training_name}.json'.format

    _model_map = {
        'gp_fitting_gaussian': GPFittingGaussian,
    }

    @classmethod
    def _get_filename(cls, model_type, problem_name, type_kernel, training_name):
        """

        :param model_type:
        :param problem_name: str
        :param type_kernel: [(str)] Must be in possible_kernels
        :param training_name: (str), prefix used to save the training data
        :return: str
        """

        kernel_name = ''
        for kernel in type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0 : -1]

        return cls._filename(
            model_type=model_type.__name__,
            problem_name=problem_name,
            type_kernel=kernel_name,
            training_name=training_name
        )

    @classmethod
    def get_gp(cls, name_model, problem_name, type_kernel, dimensions, bounds, type_bounds=None,
               n_training=0, noise=False, training_data=None, points=None, training_name=None,
               mle=True, thinning=0, n_samples=None, random_seed=DEFAULT_RANDOM_SEED):
        """
        Fetch a GP model from file if it exists, otherwise train a new model and save it locally.

        :param name_model: str
        :param problem_name: str
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param bounds:  [([float, float] or [float])], the first case is when the bounds are
            lower or upper bound of the respective entry; in the second case, it's list of finite
            points representing the domain of that entry.
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param n_training: int
        :param noise: (boolean) If true, we get noisy evaluations.
        :param training_data: {'points': [[float]], 'evaluations': [float],
            'var_noise': [float] or None}
        :param points: [[float]]. If training_data is None, we can evaluate the objective
            function in these points.
        :param training_name: (str), prefix used to save the training data.
        :param mle: (boolean) If true, fits the GP by MLE.
        :param thinning: (int)
        :param n_samples: (int) If the objective is noisy, we take n_samples of the function to
            estimate its value.
        :param random_seed: (int)

        :return: (GPFittingGaussian) - An instance of GPFittingGaussian
        """
        model_type = cls._model_map[name_model]

        if training_name is None:
            training_name = 'default_training_data_%d_points_rs_%d' % (n_training, random_seed)

        f_name = cls._get_filename(model_type, problem_name, type_kernel, training_name)
        gp_path = path.join(GP_DIR, f_name)

        data = JSONFile.read(gp_path)

        if data is not None:
            return model_type.deserialize(data)

        if training_data is None:
            training_data = TrainingDataService.get_training_data(problem_name, training_name,
                                                                  bounds,
                                                                  n_training=n_training,
                                                                  points=points,
                                                                  noise=noise,
                                                                  n_samples=n_samples,
                                                                  random_seed=random_seed,
                                                                  type_bounds=type_bounds)

        logger.info("Training %s" % model_type.__name__)

        gp_model = model_type.train(type_kernel, dimensions, mle, training_data, bounds,
                                    thinning=thinning, random_seed=random_seed)

        JSONFile.write(gp_model.serialize(), gp_path)

        return gp_model
