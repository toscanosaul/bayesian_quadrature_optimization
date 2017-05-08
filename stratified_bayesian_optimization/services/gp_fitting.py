from __future__ import absolute_import

from os import path

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.lib.constant import GP_DIR
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.models.gp_fitting_gaussian import GPFittingGaussian

logger = SBOLog(__name__)


class GPFittingService(object):
    _filename = 'gp_{problem_name}_{type_kernel}_{n_training}.json'.format

    _model_map = {
        'gp_fitting_gaussian': GPFittingGaussian,
    }

    @classmethod
    def _get_filename(cls, model_type, problem_name, type_kernel, n_training):
        """

        :param model_type:
        :param problem_name: str
        :param type_kernel: [(str)] Must be in possible_kernels
        :param n_training: (int) Number of training points
        :return: str
        """
        return cls._filename(
            model_type=model_type.__name__,
            problem_name=problem_name,
            type_kernel=type_kernel,
            n_training=n_training
        )

    @classmethod
    def get_gp(cls, name_model, problem_name, type_kernel, dimensions, training_data,
               mle=True, thinning=0):
        """
        Fetch a GP model from file if it exists, otherwise train a new model and save it locally.

        :param name_model: str
        :param problem_name: str
        :param type_kernel: [(str)] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL]
        :param training_data: {'points': np.array(nxm), 'evaluations': np.array(n),
            'var_noise': np.array(n) or None}
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product
        :param mle: (boolean) If true, fits the GP by MLE.
        :param thinning: (int)

        :return: (GPFittingGaussian) - An instance of GPFittingGaussian
        """
        model_type = cls._model_map[name_model]

        n_training = len(training_data['evaluations'])
        f_name = cls._get_filename(model_type, problem_name, type_kernel, n_training)
        gp_path = path.join(GP_DIR, f_name)

        data = JSONFile.read(gp_path)
        if data is not None:
            return model_type.deserialize(data)

        logger.info("Training %s" % model_type.__name__)
        gp_model = model_type.train(problem_name, type_kernel, n_training, dimensions,
                                    mle, thinning)

        JSONFile.write(gp_model.serialize(), gp_path)

        return gp_model

    type_kernel, training_data, dimensions, mle, thinning = 0
