from __future__ import absolute_import

import numpy as np


class BayesianEvaluations(object):

    @classmethod
    def evaluate(cls, function, point, gp_model, number_samples=20, random_seed=None, *args,
                 **kwargs):
        """
        Estimates E[function(point | parameters)] where the expectation is over the parameters of
        the gp_model.

        :param function:
        :param point: np.array(n)
        :param gp_model: GP-model
        :param number_samples: int
        :param random_seed: int
        :param args: additional arguments for the function
        :param kwargs: additional arguments for the function
        :return: float
        """

        if random_seed is not None:
            np.random.seed(random_seed)
        # TODO: BEFORE CALLING THIS FUNCTION WE SHOULD START A NEW CHAIN WITH ENOUGH SAMPLES.

        parameters = gp_model.samples_parameters[-number_samples:]

        values = []

        for parameter in parameters:
            parameter = [parameter[0], parameter[1], parameter[2:]]
            values.append(function(point, *(args + tuple(parameter)), **kwargs))

        if type(values[0]) == float:
            value = np.mean(values)
            std = np.std(values)
        else:
            value = np.mean(values, axis=0)
            std = np.std(values, axis=0)

        return value, std
