from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.services.domain import (
    DomainService
)
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.lib.constant import (
    SBO,
)
from stratified_bayesian_optimization.entities.objective import Objective

logger = SBOLog(__name__)


class BGO(object):
    _possible_optimization_methods = [SBO]

    @classmethod
    def from_spec(cls, spec):
        """
        Construct BGO instance from spec
        :param spec: RunSpecEntity

        :return: BGO
        # TO DO: It now only returns domain
        """
        logger.info("Training GP model")

        gp_model = GPFittingService.from_dict(spec)
        quadrature = None

        method_optimization = spec.get('method_optimization')

        if method_optimization not in cls._possible_optimization_methods:
            raise Exception("Incorrect BGO method")

        if method_optimization == SBO:
            x_domain = spec.get('x_domain')
            distribution = spec.get('distribution')
            parameters_distribution = spec.get('parameters_distribution')
            quadrature = BayesianQuadrature(gp_model, x_domain, distribution,
                                            parameters_distribution=parameters_distribution)

        acquistion_funciton = None

        bgo = cls(acquistion_funciton, gp_model, quadrature)

        return bgo

    def __init__(self, acquisition_function, gp_model, quadrature=None):
        self.acquisition_function = acquisition_function
        self.gp_model = gp_model
        self.quadrature = quadrature
        self.objective = Objective({'evaluated_points': [], 'objective_values': [],
                                    'standard_deviation_evaluations': []})

    def optimize(self, domain, function, n_iterations=5, random_seed=None, minimize=False):
        """
        Optimize objective over the domain.
        :param domain: (DomainEntity)
        :param function:
        :param n_iterations: int
        :param random_seed: int
        :param minimize: (boolean) the function is minimized if minimized is True.

        :return: Objective
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        optimize_mean = self.quadrature.optimize_posterior_mean(minimize=minimize)
        print optimize_mean
        return {}

    @classmethod
    def run_spec(cls, spec):
        """
        Run spec file

        :param spec: RunSpecEntity
        :return: {
            'number_iterations': int,
            'optimal_points': {
                (int) iteration: [float]
            },
            'optimal_value': {
                (int) iteration: float
            }
        }
        """
        bgo = cls.from_spec(spec)
        domain = DomainService.from_dict(spec)

        result = bgo.optimize(domain, 0)
        return result
