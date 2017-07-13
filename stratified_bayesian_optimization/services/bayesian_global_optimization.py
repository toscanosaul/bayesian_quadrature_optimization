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

        acquistion_function = None

        problem_name = spec.get('problem_name')
        training_name = spec.get('training_name')
        random_seed = spec.get('random_seed')
        n_training = spec.get('n_training')
        n_samples = spec.get('n_samples')
        noise = spec.get('noise')
        minimize = spec.get('minimize')
        n_iterations = spec.get('n_iterations')
        name_model = spec.get('name_model')

        bgo = cls(acquistion_function, gp_model, n_iterations, problem_name, training_name,
                  random_seed, n_training, name_model, method_optimization, minimize=minimize,
                  n_samples=n_samples, noise=noise, quadrature=quadrature)

        return bgo

    def __init__(self, acquisition_function, gp_model, n_iterations, problem_name, training_name,
                 random_seed, n_training, name_model, method_optimization, minimize=False,
                 n_samples=None, noise=False, quadrature=None):

        self.acquisition_function = acquisition_function
        self.gp_model = gp_model
        self.method_optimization = method_optimization
        self.quadrature = quadrature
        self.problem_name = problem_name
        self.training_name = training_name
        self.name_model = name_model
        self.objective = Objective(problem_name, training_name, random_seed, n_training, n_samples,
                                   noise)
        self.n_iterations = n_iterations
        self.minimize = minimize

    def optimize(self, random_seed=None):
        """
        Optimize objective over the domain.
        :param random_seed: int

        :return: Objective
        """
        if random_seed is not None:
            np.random.seed(random_seed)


        if self.method_optimization == SBO:
            model = self.quadrature

        optimize_mean = model.optimize_posterior_mean(minimize=self.minimize)
        optimal_value = \
            self.objective.add_point(optimize_mean['solution'], optimize_mean['optimal_value'][0])

        model.write_debug_data(self.problem_name, self.name_model, self.training_name)
        return {
            'optimal_solution': optimize_mean['solution'],
            'optimal_value': optimal_value,

        }

    @classmethod
    def run_spec(cls, spec):
        """
        Run spec file

        :param spec: RunSpecEntity
        :return: {
            'optimal_value': float,
            'optimal_solution': np.array(n),
        }
        """
        bgo = cls.from_spec(spec)
        domain = DomainService.from_dict(spec)

        # WE CAN STILL ADD THE DOMAIN IF NEEDED FOR THE KG
        result = bgo.optimize()
        return result
