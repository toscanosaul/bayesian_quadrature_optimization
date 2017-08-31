from __future__ import absolute_import

import numpy as np

from stratified_bayesian_optimization.services.domain import (
    DomainService
)
from stratified_bayesian_optimization.services.gp_fitting import GPFittingService
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.numerical_tools.bayesian_quadrature import BayesianQuadrature
from stratified_bayesian_optimization.lib.constant import (
    SBO_METHOD,
    MULTI_TASK_METHOD,
    TASKS,
    DOGLEG,
    LBFGS_NAME,
    SGD_NAME,
)
from stratified_bayesian_optimization.entities.objective import Objective
from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.acquisition_functions.multi_task import MultiTasks
from stratified_bayesian_optimization.services.training_data import TrainingDataService

logger = SBOLog(__name__)


class BGO(object):
    _possible_optimization_methods = [SBO_METHOD, MULTI_TASK_METHOD]

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
        acquisition_function = None

        method_optimization = spec.get('method_optimization')

        domain = DomainService.from_dict(spec)

        if method_optimization not in cls._possible_optimization_methods:
            raise Exception("Incorrect BGO method")

        if method_optimization == SBO_METHOD:
            x_domain = spec.get('x_domain')
            distribution = spec.get('distribution')
            parameters_distribution = spec.get('parameters_distribution')
            quadrature = BayesianQuadrature(gp_model, x_domain, distribution,
                                            parameters_distribution=parameters_distribution)

            acquisition_function = SBO(quadrature, np.array(domain.discretization_domain_x))
        elif method_optimization == MULTI_TASK_METHOD:
            x_domain = spec.get('x_domain')
            distribution = spec.get('distribution')
            parameters_distribution = spec.get('parameters_distribution')
            quadrature = BayesianQuadrature(gp_model, x_domain, distribution,
                                            parameters_distribution=parameters_distribution,
                                            model_only_x=True)
            acquisition_function = MultiTasks(quadrature,
                                             quadrature.parameters_distribution.get(TASKS))


        problem_name = spec.get('problem_name')
        training_name = spec.get('training_name')
        random_seed = spec.get('random_seed')
        n_samples = spec.get('n_samples')
        noise = spec.get('noise')
        minimize = spec.get('minimize')
        n_iterations = spec.get('n_iterations')
        name_model = spec.get('name_model')
        parallel = spec.get('parallel')
        n_training = spec.get('n_training')
        number_points_each_dimension_debug = spec.get('number_points_each_dimension_debug')
        n_samples_parameters = spec.get('n_samples_parameters', 0)
        use_only_training_points = spec.get('use_only_training_points', True)


        bgo = cls(acquisition_function, gp_model, n_iterations, problem_name, training_name,
                  random_seed, n_training, name_model, method_optimization, minimize=minimize,
                  n_samples=n_samples, noise=noise, quadrature=quadrature, parallel=parallel,
                  number_points_each_dimension_debug=number_points_each_dimension_debug,
                  n_samples_parameters=n_samples_parameters,
                  use_only_training_points=use_only_training_points)

        return bgo

    def __init__(self, acquisition_function, gp_model, n_iterations, problem_name, training_name,
                 random_seed, n_training, name_model, method_optimization, minimize=False,
                 n_samples=None, noise=False, quadrature=None, parallel=True,
                 number_points_each_dimension_debug=None, n_samples_parameters=0,
                 use_only_training_points=True):

        self.acquisition_function = acquisition_function
        self.acquisition_function.args_handler = (True, name_model, problem_name, training_name,
                                                  n_training, random_seed, n_samples_parameters)

        self.acquisition_function.clean_cache()
        self.gp_model = gp_model

        self.method_optimization = method_optimization
        self.quadrature = quadrature

        if quadrature is not None:
            self.quadrature.args_handler = (True, name_model, problem_name, training_name,
                                                  n_training, random_seed, n_samples_parameters)

        self.problem_name = problem_name
        self.training_name = training_name
        self.name_model = name_model
        self.objective = Objective(problem_name, training_name, random_seed, n_training, n_samples,
                                   noise, self.method_optimization, n_samples_parameters)

        if not use_only_training_points:
            self.objective.set_data_from_file()

        self.n_iterations = n_iterations
        self.minimize = minimize
        self.parallel = parallel
        self.n_training = n_training
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.number_points_each_dimension_debug = number_points_each_dimension_debug

    def optimize(self, random_seed=None, start=None, debug=False, monte_carlo_sbo=False,
                 n_samples_mc=1, n_restarts_mc=1, n_best_restarts_mc=0,
                 n_restarts=10, n_best_restarts=0, n_samples_parameters=0, n_restarts_mean=1000,
                 n_best_restarts_mean=100, method_opt_mc=None, maxepoch=10,
                 n_samples_parameters_mean=0, maxepoch_mean=20, threshold_sbo=None,
                 **opt_params_mc):
        """
        Optimize objective over the domain.
        :param random_seed: int
        :param start: (np.array(n)) starting point for the optimization of VOI
        :param debug: (boolean) If true, saves evaluations of the VOI and posterior mean at each
            iteration.
        :param monte_carlo_sbo: (boolean) If True, estimates the objective function and gradient by
            MC.
        :param n_samples_mc: (int) Number of samples for the MC method.
        :param n_restarts_mc: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param n_best_restarts_mc: (int) Number of best restarting points chosen to optimize
            a_{n+1} given a sample.
        :param n_restarts: (int) Number of restarts of the VOI
        :param n_best_restarts: (int) Number of best restarting points chosen to optimize the VOI
        :param n_samples_parameters: (int)
        :param n_restarts_mean: int
        :param n_best_restarts_mean: int
        :param method_opt_mc: (str)
        :param maxepoch: (int) For SGD
        :param n_samples_parameters_mean: (int)
        :param maxepoch_mean: (int)
        :param threshold_sbo: (float) If VOI < threshold_sbo, then we choose randomly a point
            instead.
        :param opt_params_mc:
            -'factr': int
            -'maxiter': int

        :return: Objective
        """

        if n_samples_parameters > 0 and n_samples_parameters_mean == 0:
            n_samples_parameters_mean = n_samples_parameters

        if method_opt_mc is None:
            method_opt_mc = LBFGS_NAME

        if random_seed is not None:
            np.random.seed(random_seed)

        threshold_af = None
        if self.method_optimization == SBO_METHOD:
            threshold_af = threshold_sbo

        if self.method_optimization == SBO_METHOD or self.method_optimization == MULTI_TASK_METHOD:
            model = self.quadrature

        noise = None

        if n_samples_parameters_mean > 0:
            method_opt_mu = SGD_NAME
        else:
            method_opt_mu = DOGLEG

        optimize_mean = model.optimize_posterior_mean(
            minimize=self.minimize, n_restarts=n_restarts_mean,
            n_best_restarts=n_best_restarts_mean, n_samples_parameters=n_samples_parameters_mean,
            start_new_chain=True, method_opt=method_opt_mu, maxepoch=maxepoch_mean)

        optimal_value = \
            self.objective.add_point(optimize_mean['solution'], optimize_mean['optimal_value'][0])

        model.write_debug_data(self.problem_name, self.name_model, self.training_name,
                               self.n_training, self.random_seed, self.method_optimization,
                               n_samples_parameters)

        if debug:
            model.generate_evaluations(
                self.problem_name, self.name_model, self.training_name, self.n_training,
                self.random_seed, 0, n_points_by_dimension=self.number_points_each_dimension_debug)

        for iteration in xrange(self.n_iterations):

            new_point_sol = self.acquisition_function.optimize(
                parallel=self.parallel, start=start, monte_carlo=monte_carlo_sbo,
                n_samples=n_samples_mc, n_restarts_mc=n_restarts_mc,
                n_best_restarts_mc=n_best_restarts_mc, n_restarts=n_restarts,
                n_best_restarts=n_best_restarts, n_samples_parameters=n_samples_parameters,
                start_new_chain=False, method_opt_mc=method_opt_mc, maxepoch=maxepoch,
                **opt_params_mc)

            value_sbo = new_point_sol['optimal_value']
            new_point = new_point_sol['solution']

            if threshold_af is not None and value_sbo < threshold_af:
                #TODO: FINISH THIS
                new_point = self.acquisition_function.random_point_domain(1)

            self.acquisition_function.write_debug_data(self.problem_name, self.name_model,
                                                       self.training_name, self.n_training,
                                                       self.random_seed,
                                                       n_samples_parameters=n_samples_parameters,
                                                       monte_carlo=monte_carlo_sbo)

            if debug:
                self.acquisition_function.generate_evaluations(
                    self.problem_name, self.name_model, self.training_name, self.n_training,
                    self.random_seed, iteration,
                    n_points_by_dimension=self.number_points_each_dimension_debug,
                    monte_carlo=monte_carlo_sbo, n_samples=n_samples_mc,
                    n_restarts_mc=n_restarts_mc)


            self.acquisition_function.clean_cache()

            evaluation = TrainingDataService.evaluate_function(self.objective.module, new_point,
                                                               self.n_samples)

            if self.objective.noise:
                noise = np.array([evaluation[1]])

            self.gp_model.add_points_evaluations(new_point.reshape((1, len(new_point))),
                                                 np.array([evaluation[0]]),
                                                 var_noise_eval=noise)

            GPFittingService.write_gp_model(self.gp_model, method=self.method_optimization,
                                            n_samples_parameters=n_samples_parameters)

            optimize_mean = model.optimize_posterior_mean(
                minimize=self.minimize, n_restarts=n_restarts_mean,
                n_best_restarts=n_best_restarts_mean,
                n_samples_parameters=n_samples_parameters_mean,
                start_new_chain=True, method_opt=method_opt_mu, maxepoch=maxepoch_mean
            )

            optimal_value = \
                self.objective.add_point(optimize_mean['solution'],
                                         optimize_mean['optimal_value'][0])

            model.write_debug_data(self.problem_name, self.name_model, self.training_name,
                                   self.n_training, self.random_seed, self.method_optimization,
                                   n_samples_parameters)

            if debug:
                model.generate_evaluations(
                    self.problem_name, self.name_model, self.training_name, self.n_training,
                    self.random_seed, iteration + 1,
                    n_points_by_dimension=self.number_points_each_dimension_debug)

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
        debug = spec.get('debug')
        monte_carlo_sbo = spec.get('monte_carlo_sbo')
        n_samples_mc = spec.get('n_samples_mc')
        n_restarts_mc = spec.get('n_restarts_mc')
        n_best_restarts_mc = spec.get('n_best_restarts_mc')
        n_best_restarts = spec.get('n_best_restarts')

        n_restarts_mean = spec.get('n_restarts_mean', 1000)
        n_best_restarts_mean = spec.get('n_best_restarts_mean', 100)

        opt_params_mc = {}
        factr = spec.get('factr_mc')
        maxiter = spec.get('maxiter_mc')
        n_restarts = spec.get('n_restarts', 10)

        n_samples_parameters = spec.get('n_samples_parameters', 0)

        if factr is not None:
            opt_params_mc['factr'] = factr
        if maxiter is not None:
            opt_params_mc['maxiter'] = maxiter

        method_opt_mc = spec.get('method_opt_mc')
        maxepoch = spec.get('maxepoch')

        n_samples_parameters_mean = spec.get('n_samples_parameters_mean', 15)
        maxepoch_mean = spec.get('maxepoch_mean', 15)


        # WE CAN STILL ADD THE DOMAIN IF NEEDED FOR THE KG
        result = bgo.optimize(debug=debug, n_samples_mc=n_samples_mc, n_restarts_mc=n_restarts_mc,
                              n_best_restarts_mc=n_best_restarts_mc,
                              monte_carlo_sbo=monte_carlo_sbo, n_restarts=n_restarts,
                              n_best_restarts=n_best_restarts,
                              n_samples_parameters=n_samples_parameters,
                              n_restarts_mean=n_restarts_mean,
                              n_best_restarts_mean=n_best_restarts_mean,
                              random_seed=bgo.random_seed, method_opt_mc=method_opt_mc,
                              n_samples_parameters_mean=n_samples_parameters_mean,
                              maxepoch_mean=maxepoch_mean,
                              maxepoch=maxepoch, **opt_params_mc)
        return result
