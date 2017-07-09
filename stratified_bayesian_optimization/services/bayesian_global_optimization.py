from __future__ import absolute_import

from stratified_bayesian_optimization.services.domain import (
    DomainService
)
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.initializers.log import SBOLog

logger = SBOLog(__name__)


class BGO(object):

    @classmethod
    def from_spec(cls, spec):
        """
        Construct BGO instance from spec
        :param spec: RunSpecEntity

        :return: BGO
        # TO DO: It now only returns domain
        """
        logger.info("Training GP model")

        training_data = TrainingDataService.from_dict(spec)
        assert training_data
        # method_optimization = spec.method_optimization

        gp_model = None
        acquistion_funciton = None

        bgo = cls(acquistion_funciton, gp_model)

        return bgo

    def __init__(self, acquisition_function, gp_model):
        self.acquisition_function = acquisition_function
        self.gp_model = gp_model

    def optimize(self, domain, function):
        """
        Optimize objective over the domain.
        :param domain: (DomainEntity)
        :param function:
        :return: Objective
        """
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
