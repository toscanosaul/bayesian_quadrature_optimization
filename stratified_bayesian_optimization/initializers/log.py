from __future__ import absolute_import

import logging
import sys

from os import path
import os

from stratified_bayesian_optimization.lib.constant import (
    LOG_DIR,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SBOLog(object):
    _filename = 'log_bgo_{name}_{model_type}_{problem_name}_{type_kernel}_{training_name}_' \
                '{n_training}_{random_seed}_samples_params_' \
                '{n_samples_parameters}.log'.format


    def __init__(self, name):
        self.name = name
        self._log = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        """
        :param msg: dict
        """

        self._log.info(msg, *args, **kwargs)

    def add_file_to_log(self, model_type, problem_name, type_kernel, training_name, n_training,
                        random_seed, n_samples_parameters):
        """
        Adds file to write the log messages

        :param model_type:
        :param problem_name:
        :param type_kernel:
        :param training_name:
        :param n_training:
        :param random_seed:
        :param n_samples_parameters:
        :return:
        """
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

        log_dir = path.join(LOG_DIR, problem_name)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        filename = self._filename(
            name=self.name,
            model_type=model_type, problem_name=problem_name, type_kernel=type_kernel,
            training_name=training_name, n_training=n_training, random_seed=random_seed,
            n_samples_parameters=n_samples_parameters)

        log_path = path.join(log_dir, filename)
        hdlr = logging.FileHandler(log_path, mode='w')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self._log.addHandler(hdlr)
