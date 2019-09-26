from __future__ import absolute_import

import numpy as np

from collections import Counter

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
    EI_METHOD,
    SDE_METHOD,
)
from stratified_bayesian_optimization.lib.distances import Distances
from stratified_bayesian_optimization.entities.objective import Objective
from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.acquisition_functions.ei import EI
from stratified_bayesian_optimization.acquisition_functions.multi_task import MultiTasks
from stratified_bayesian_optimization.services.training_data import TrainingDataService
from stratified_bayesian_optimization.acquisition_functions.sde import SDE
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class BGO(object):
    _possible_optimization_methods = [SBO_METHOD, MULTI_TASK_METHOD, EI_METHOD, SDE_METHOD]

    @classmethod
    def from_spec(cls, spec):
        """
        Construct BGO instance from spec
        :param spec: RunSpecEntity

        :return: BGO
        # TO DO: It now only returns domain
        """

        random_seed = spec.get('random_seed')
        method_optimization = spec.get('method_optimization')

        logger.info("Training GP model")
        logger.info("Random seed is: %d" % random_seed)
        logger.info("Algorithm used is:")
        logger.info(method_optimization)

        gp_model = GPFittingService.from_dict(spec)

        simplex_domain = spec.get('simplex_domain', None)

        noise = spec.get('noise')
        quadrature = None
        acquisition_function = None

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
                                            model_only_x=True, tasks=True)
            acquisition_function = MultiTasks(quadrature,
                                             quadrature.n_tasks)
        elif method_optimization == EI_METHOD:
            acquisition_function = EI(gp_model, noisy_evaluations=noise)
        elif method_optimization == SDE_METHOD:
            x_domain = len(spec.get('x_domain'))
            parameters_distribution = spec.get('parameters_distribution')
            domain_random = np.array(parameters_distribution['domain_random'])
            weights = np.array(parameters_distribution['weights'])
            acquisition_function = SDE(gp_model, domain_random, x_domain, weights)

        problem_name = spec.get('problem_name')
        training_name = spec.get('training_name')
        n_samples = spec.get('n_samples')
        minimize = spec.get('minimize')
        n_iterations = spec.get('n_iterations')
        name_model = spec.get('name_model')
        parallel = spec.get('parallel')
        n_training = spec.get('n_training')
        number_points_each_dimension_debug = spec.get('number_points_each_dimension_debug')
        n_samples_parameters = spec.get('n_samples_parameters', 0)
        use_only_training_points = spec.get('use_only_training_points', True)


        n_iterations = n_iterations - (len(gp_model.training_data['evaluations']) - n_training)

        bgo = cls(acquisition_function, gp_model, n_iterations, problem_name, training_name,
                  random_seed, n_training, name_model, method_optimization, minimize=minimize,
                  n_samples=n_samples, noise=noise, quadrature=quadrature, parallel=parallel,
                  number_points_each_dimension_debug=number_points_each_dimension_debug,
                  n_samples_parameters=n_samples_parameters,
                  use_only_training_points=use_only_training_points)

        if n_training < len(bgo.gp_model.training_data['evaluations']):
            extra_iterations = len(bgo.gp_model.training_data['evaluations']) - n_training
            data = JSONFile.read(bgo.objective.file_path)
            bgo.objective.evaluated_points = data['evaluated_points'][0:extra_iterations]
            bgo.objective.objective_values = data['objective_values'][0:extra_iterations]
            bgo.objective.model_objective_values = \
                data['model_objective_values'][0:extra_iterations]
            bgo.objective.standard_deviation_evaluations = data['standard_deviation_evaluations']



        return bgo

    def __init__(self, acquisition_function, gp_model, n_iterations, problem_name, training_name,
                 random_seed, n_training, name_model, method_optimization, minimize=False,
                 n_samples=None, noise=False, quadrature=None, parallel=True,
                 number_points_each_dimension_debug=None, n_samples_parameters=0,
                 use_only_training_points=True, objective_function=None, training_function=None):

        self.acquisition_function = acquisition_function

        kernel_name = ''
        for kernel in gp_model.type_kernel:
            kernel_name += kernel + '_'
        kernel_name = kernel_name[0: -1]
        self.acquisition_function.args_handler = (True, name_model, problem_name, kernel_name,
                                                  training_name, n_training, random_seed,
                                                  n_samples_parameters)
        self.acquisition_function.clean_cache()
        self.gp_model = gp_model

        self.method_optimization = method_optimization
        self.quadrature = quadrature

        if quadrature is not None:
            self.quadrature.args_handler = (True, name_model, problem_name, kernel_name,
                                            training_name, n_training, random_seed,
                                            n_samples_parameters)

        self.problem_name = problem_name
        self.training_name = training_name
        self.name_model = name_model
        self.objective = Objective(problem_name, training_name, random_seed, n_training, n_samples,
                                   noise, self.method_optimization, n_samples_parameters,
                                   objective_function=objective_function,
                                   training_function=training_function)

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
                 optimize_only_posterior_mean=False, start_optimize_posterior_mean=0,
                 optimize_mean_each_iteration=True, default_n_samples_parameters=None,
                 default_n_samples=None, **opt_params_mc):
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

        :return: {
            'optimal_solution': optimize_mean['solution'],
            'optimal_value': optimal_value,
        }
        """

        if optimize_only_posterior_mean:
            # only for noisless problems
            chosen_points = self.gp_model.data.copy()
            n_training = self.n_training
            start_optimize_posterior_mean = np.min(len(chosen_points['evaluations']) - n_training,
                                                   start_optimize_posterior_mean)
            total_points = \
                len(chosen_points['evaluations']) - n_training - start_optimize_posterior_mean
            self.gp_model.clean_cache()
            self.gp_model.data['evaluations'] = \
                self.gp_model.data['evaluations'][0: n_training + start_optimize_posterior_mean]
            self.gp_model.data['points'] =\
                self.gp_model.data['points'][0: n_training + start_optimize_posterior_mean, :]

            self.objective.evaluated_points = \
                self.objective.evaluated_points[0:start_optimize_posterior_mean]
            self.objective.objective_values = \
                self.objective.objective_values[0:start_optimize_posterior_mean]
            self.objective.model_objective_values = \
                self.objective.model_objective_values[0:start_optimize_posterior_mean]

        start_ei = True
        if self.quadrature is not None and self.quadrature.task_continue:
            start_ei = False

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
        else:
            model = self.gp_model

        noise = None

        if n_samples_parameters_mean > 0:
            method_opt_mu = SGD_NAME
        else:
            method_opt_mu = DOGLEG

        if optimize_mean_each_iteration or 0 == self.n_iterations:
            if self.method_optimization == SDE_METHOD:
                optimize_mean = self.acquisition_function.optimize_mean(
                    n_restarts=n_restarts_mean,
                    candidate_solutions=self.objective.evaluated_points,
                    candidate_values=self.objective.objective_values)
            else:
                optimize_mean = model.optimize_posterior_mean(
                    minimize=self.minimize, n_restarts=n_restarts_mean,
                    n_best_restarts=n_best_restarts_mean,
                    n_samples_parameters=n_samples_parameters_mean,
                    start_new_chain=True, method_opt=method_opt_mu, maxepoch=maxepoch_mean,
                    candidate_solutions=self.objective.evaluated_points,
                    candidate_values=self.objective.objective_values)

            optimal_value = \
                self.objective.add_point(
                    optimize_mean['solution'], optimize_mean['optimal_value'][0])

            model.write_debug_data(self.problem_name, self.name_model, self.training_name,
                                   self.n_training, self.random_seed, self.method_optimization,
                                   n_samples_parameters)

        if debug:
            model.generate_evaluations(
                self.problem_name, self.name_model, self.training_name, self.n_training,
                self.random_seed, 0, n_points_by_dimension=self.number_points_each_dimension_debug)

        start_new_chain_acquisition_function = False
        if optimize_mean_each_iteration:
            start_new_chain_acquisition_function = True

        for iteration in range(self.n_iterations):
            evaluation = None
            if not optimize_only_posterior_mean or iteration >= total_points:
                new_point_sol = self.acquisition_function.optimize(
                    parallel=self.parallel, start=start, monte_carlo=monte_carlo_sbo,
                    n_samples=n_samples_mc, n_restarts_mc=n_restarts_mc,
                    n_best_restarts_mc=n_best_restarts_mc, n_restarts=n_restarts,
                    n_best_restarts=n_best_restarts, n_samples_parameters=n_samples_parameters,
                    start_new_chain=start_new_chain_acquisition_function,
                    method_opt_mc=method_opt_mc, maxepoch=maxepoch, start_ei=start_ei,
                    default_n_samples_parameters=default_n_samples_parameters,
                    default_n_samples=default_n_samples, **opt_params_mc)
            else:
                point = \
                    chosen_points['points'][n_training + start_optimize_posterior_mean + iteration, :]
                new_point_sol = {'optimal_value': 0.0, 'solution': point}
                evaluation = \
                    chosen_points['evaluations'][n_training + start_optimize_posterior_mean + iteration]
                evaluation = [evaluation]

            value_sbo = new_point_sol['optimal_value']
            new_point = new_point_sol['solution']

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

            if evaluation is None:
                if self.objective.module is not None:
                    evaluation = TrainingDataService.evaluate_function(
                        self.objective.module, new_point, self.n_samples)
                else:
                    if self.n_samples == 0 or self.n_samples is None:
                        evaluation = self.objective.training_function(new_point)
                    else:
                        evaluation = self.objective.training_function(new_point, self.n_samples)

            if self.objective.noise:
                noise = np.array([evaluation[1]])

            self.gp_model.add_points_evaluations(new_point.reshape((1, len(new_point))),
                                                 np.array([evaluation[0]]),
                                                 var_noise_eval=noise)

            GPFittingService.write_gp_model(self.gp_model, method=self.method_optimization,
                                            n_samples_parameters=n_samples_parameters)

            if optimize_mean_each_iteration or iteration == self.n_iterations - 1:
                if self.method_optimization == SDE_METHOD:
                    optimize_mean = self.acquisition_function.optimize_mean(
                        n_restarts=n_restarts_mean,
                        candidate_solutions=self.objective.evaluated_points,
                        candidate_values=self.objective.objective_values)
                else:
                    optimize_mean = model.optimize_posterior_mean(
                        minimize=self.minimize, n_restarts=n_restarts_mean,
                        n_best_restarts=n_best_restarts_mean,
                        n_samples_parameters=n_samples_parameters_mean,
                        start_new_chain=True, method_opt=method_opt_mu, maxepoch=maxepoch_mean,
                        candidate_solutions=self.objective.evaluated_points,
                        candidate_values=self.objective.objective_values
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
      #  spec.simplex_domain = None
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

        threshold_sbo = spec.get('threshold_sbo')

        optimize_only_posterior_mean = spec.get('optimize_only_posterior_mean', False)
        start_optimize_posterior_mean = spec.get('start_optimize_posterior_mean', 0)

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
                              maxepoch=maxepoch, threshold_sbo=threshold_sbo,
                              optimize_only_posterior_mean=optimize_only_posterior_mean,
                              start_optimize_posterior_mean=start_optimize_posterior_mean,
                              **opt_params_mc)
        return result
