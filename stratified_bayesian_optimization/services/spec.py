from __future__ import absolute_import

from os import path
import os

import numpy as np

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    DEFAULT_RANDOM_SEED,
    UNIFORM_FINITE,
    DOGLEG,
    PROBLEM_DIR,
    PARTIAL_RESULTS,
    AGGREGATED_RESULTS,
)

logger = SBOLog(__name__)


class SpecService(object):

    _filename_results = 'results_{problem_name}_{training_name}_{n_points}_{random_seed}_' \
                        '{method}_samples_params_{n_samples_parameters}.json'.format

    _aggregated_results = 'results_{problem_name}_{training_name}_{n_points}_{method}.json'.format

    @classmethod
    def generate_dict_spec(cls, problem_name, dim_x, bounds_domain_x, training_name, type_kernel,
                           dimensions, bounds_domain=None, number_points_each_dimension=None,
                           choose_noise=True, method_optimization='sbo', type_bounds=None,
                           n_training=10, points=None, noise=False, n_samples=0,
                           random_seed=DEFAULT_RANDOM_SEED, parallel=True,
                           name_model='gp_fitting_gaussian', mle=True, thinning=0, n_burning=0,
                           max_steps_out=1, training_data=None, x_domain=None, distribution=None,
                           parameters_distribution=None, minimize=False, n_iterations=5,
                           kernel_values=None, mean_value=None, var_noise_value=None,
                           debug=False, same_correlation=False,
                           number_points_each_dimension_debug=None, monte_carlo_sbo=False,
                           n_samples_mc=1, n_restarts_mc=1, factr_mc=1e12, maxiter_mc=1000,
                           use_only_training_points=True, n_restarts=10, n_best_restarts=10,
                           n_best_restarts_mc=1, n_samples_parameters=0, n_restarts_mean=1000,
                           n_best_restarts_mean=100, method_opt_mc=DOGLEG, maxepoch=10,
                           n_samples_parameters_mean=15, maxepoch_mean=20, threshold_sbo=None,
                           parallel_training=True):

        """
        Generate dict that represents run spec.

        :param problem_name: (str)
        :param dim_x: int
        :param bounds_domain_x: [(float, float)]
        :param training_name: (str) Prefix for the file of the training data
        :param type_kernel: [str] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL].
            If we want to use a scaled NAME_1_KERNEL, the parameter must be
            [SCALED_KERNEL, NAME_1_KERNEL].
        :param dimensions: [int]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
        :param bounds_domain: [([float, float] or [float])], the first case is when the bounds are
            lower or upper bound of the respective entry; in the second case, it's list of finite
            points representing the domain of that entry.
        :param number_points_each_dimension: [int] number of points in each dimension for the
            discretization of the domain of x.
        :param choose_noise: boolean
        :param method_optimization: (str) Options: 'SBO', 'KG'
        :param type_bounds: [0 or 1], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param n_training: (int) number of training points
        :param points: [[float]], the objective function is evaluated on these points to generate
            the training data.
        :param noise: boolean, true if the evaluations are noisy
        :param n_samples: (int),  If noise is true, we take n_samples of the function to estimate
            its value.
        :param random_seed: (int)
        :param parallel: (boolean) Train in parallel if it's True.
        :param name_model: str
        :param mle: (boolean) If true, fits the GP by MLE.
        :param thinning: (int)
        :param n_burning: (int) Number of burnings samples for the MCMC.
        :param max_steps_out: (int)  Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param training_data: {'points': [[float]], 'evaluations': [float],
            'var_noise': [float] or None}
        :param x_domain: [int], indices of the x domain
        :param distribution: (str), probability distribution for the Bayesian quadrature
        :param parameters_distribution: {str: float}, parameters of the distribution
        :param minimize: (boolean) Minimizes the function if minimize is True.
        :param n_iterations: (int)
        :param kernel_values: [float], contains the default values of the parameters of the kernel
        :param mean_value: [float], It contains the value of the mean parameter.
        :param var_noise_value: [float], It contains the variance of the noise of the model
        :param debug: (boolean) If true, it generates the evaluations of the VOI and posterior mean.
        :param same_correlation: (boolean) If true, it uses the same correlations for the task
            kernel.
        :param number_points_each_dimension_debug: ([int]) Number of points for the discretization
            of the debug plots.
        :param monte_carlo_sbo: (boolean) If True, estimates the objective function and gradient by
            MC.
        :param n_samples_mc: (int) Number of samples for the MC method.
        :param n_restarts_mc: (int) Number of restarts to optimize a_{n+1} given a sample.
        :param factr_mc: (float) Parameter of LBFGS to optimize a sample of SBO when using MC.
        :param maxiter_mc: (int) Max number of iterations to optimize a sample of SBO when using MC.
        :param use_only_training_points: (boolean) Uses only training points in the cached gp model
            if it's True.
        :param n_restarts: (int) Number of starting points to optimize the acquisition function
        :param n_best_restarts: (int) Number of best starting points chosen from the n_restart
            points.
        :param n_best_restarts_mc: (int) Number of best restarting points used to optimize a_{n+1}
            given a sample.
        :param n_samples_parameters: (int) Number of samples of the parameter to compute the VOI
            using a Bayesian approach.
        :param n_restarts_mean: (int) Number of starting points to optimize the posterior mean.
        :param n_best_restarts_mean: (int)
        :param method_opt_mc: (str) Optimization method when estimating SBO or KG by samples
        :param maxepoch: (int) For SGD
        :param n_samples_parameters_mean: (int)
        :param maxepoch_mean: (int) Maxepoch for the optimization of the posterior mean
        :param threshold_sbo: (int) If SBO < threshold_sbo, we randomly choose a point instead.
        :param parallel_training: (boolean) If True, the training data is computed in parallel

        :return: dict
        """

        if bounds_domain is None:
            bounds_domain = [[bound[0], bound[1]] for bound in bounds_domain_x]

        if number_points_each_dimension is None:
            number_points_each_dimension = [10] * dim_x

        if type_bounds is None:
            type_bounds = [0] * len(bounds_domain)

        if kernel_values is None:
            kernel_values = []

        if mean_value is None:
            mean_value = []

        if var_noise_value is None:
            var_noise_value = []

        if points is None:
            points = []

        if training_data is None:
            training_data = {}

        if x_domain is None:
            x_domain = []

        if distribution is None:
            distribution = UNIFORM_FINITE

        if parameters_distribution is None:
            parameters_distribution = {}

        if debug is None:
            debug = False

        if same_correlation is None:
            same_correlation = False

        return {
            'problem_name': problem_name,
            'dim_x': dim_x,
            'choose_noise': choose_noise,
            'bounds_domain_x': bounds_domain_x,
            'number_points_each_dimension': number_points_each_dimension,
            'method_optimization': method_optimization,
            'training_name': training_name,
            'bounds_domain': bounds_domain,
            'n_training': n_training,
            'points': points,
            'noise': noise,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'parallel': parallel,
            'type_bounds': type_bounds,
            'type_kernel': type_kernel,
            'dimensions': dimensions,
            'name_model': name_model,
            'mle': mle,
            'thinning': thinning,
            'n_burning': n_burning,
            'max_steps_out': max_steps_out,
            'training_data': training_data,
            'x_domain': x_domain,
            'distribution': distribution,
            'parameters_distribution': parameters_distribution,
            'minimize': minimize,
            'n_iterations': n_iterations,
            'var_noise_value': var_noise_value,
            'mean_value': mean_value,
            'kernel_values': kernel_values,
            'debug': debug,
            'same_correlation': same_correlation,
            'number_points_each_dimension_debug': number_points_each_dimension_debug,
            'monte_carlo_sbo': monte_carlo_sbo,
            'n_samples_mc': n_samples_mc,
            'n_restarts_mc': n_restarts_mc,
            'factr_mc': factr_mc,
            'maxiter_mc': maxiter_mc,
            'use_only_training_points': use_only_training_points,
            'n_restarts': n_restarts,
            'n_best_restarts': n_best_restarts,
            'n_best_restarts_mc': n_best_restarts_mc,
            'n_samples_parameters': n_samples_parameters,
            'n_restarts_mean': n_restarts_mean,
            'n_best_restarts_mean': n_best_restarts_mean,
            'method_opt_mc': method_opt_mc,
            'maxepoch': maxepoch,
            'n_samples_parameters_mean': n_samples_parameters_mean,
            'maxepoch_mean': maxepoch_mean,
            'threshold_sbo': threshold_sbo,
            'parallel_training': parallel_training,
        }

    # TODO - generate a list of runspecentities over different parameters

    @classmethod
    def generate_dict_multiple_spec(
            cls, n_specs, problem_names, dim_xs, bounds_domain_xs, training_names, type_kernels,
            dimensionss, bounds_domains=None, number_points_each_dimensions=None,
            choose_noises=None, method_optimizations=None, type_boundss=None, n_trainings=None,
            pointss=None, noises=None, n_sampless=None, random_seeds=None, parallels=None,
            name_models=None, mles=None, thinnings=None, n_burnings=None, max_steps_outs=None,
            training_datas=None, x_domains=None, distributions=None, parameters_distributions=None,
            minimizes=None, n_iterationss=None, kernel_valuess=None, mean_values=None,
            var_noise_values=None, caches=None, debugs=None, same_correlations=None,
            number_points_each_dimension_debugs=None, monte_carlo_sbos=None, n_samples_mcs=None,
            n_restarts_mcs=None, factr_mcs=None, maxiter_mcs=None, use_only_training_pointss=None,
            n_restartss=None, n_best_restartss=None, n_best_restarts_mcs=None,
            n_samples_parameterss=None, n_restarts_means=None, n_best_restarts_means=None,
            method_opt_mcs=None, maxepochs=None, n_samples_parameters_means=None,
            maxepoch_means=None, threshold_sbos=None, parallel_trainings=None,
            optimize_only_posterior_means=None, start_optimize_posterior_means=None):

        """
        Generate dict that represents multiple run specs

        :param n_specs: (int) number of specifications
        :param problem_names: [str]
        :param dim_xs: [int]
        :param bounds_domain_xs: [[(float, float)]]
        :param training_names: ([str]) Prefix for the file of the training data
        :param type_kernels: [[str]] Must be in possible_kernels. If it's a product of kernels it
            should be a list as: [PRODUCT_KERNELS_SEPARABLE, NAME_1_KERNEL, NAME_2_KERNEL].
            If we want to use a scaled NAME_1_KERNEL, the parameter must be
            [SCALED_KERNEL, NAME_1_KERNEL].
        :param dimensionss: [[int]]. It has only the n_tasks for the task_kernels, and for the
            PRODUCT_KERNELS_SEPARABLE contains the dimensions of every kernel in the product, and
            the total dimension of the product_kernels_separable too in the first entry.
        :param bounds_domains: [[([float, float] or [float])]], the first case is when the bounds
            are lower or upper bound of the respective entry; in the second case, it's list of
            finite points representing the domain of that entry.
        :param number_points_each_dimensions: [[int]] number of points in each dimension for the
            discretization of the domain of x.
        :param choose_noises: [boolean]
        :param method_optimizations: [(str)] Options: 'SBO', 'KG'
        :param type_boundss: [[0 or 1]], 0 if the bounds are lower or upper bound of the respective
            entry, 1 if the bounds are all the finite options for that entry.
        :param n_trainings: ([int]) number of training points
        :param pointss: [[[float]]], the objective function is evaluated on these points to generate
            the training data.
        :param noises: [boolean], true if the evaluations are noisy
        :param n_sampless: ([int]),  If noise is true, we take n_samples of the function to estimate
            its value.
        :param random_seeds: ([int])
        :param parallels: ([boolean]) Train in parallel if it's True.
        :param name_models: [str]
        :param mles: ([boolean]) If true, fits the GP by MLE.
        :param thinnings: [int]
        :param n_burnings: ([int]) Number of burnings samples for the MCMC.
        :param max_steps_outs: ([int])  Maximum number of steps out for the stepping out  or
                doubling procedure in slice sampling.
        :param training_datas: [{'points': [[float]], 'evaluations': [float],
            'var_noise': [float] or None}]
        :param x_domains: [[int]], indices of the x domain
        :param distributions: [str], probability distributions for the Bayesian quadrature
        :param parameters_distributions: [{str: float}], parameters of the distributions
        :param minimizes: [boolean]
        :param n_iterationss: [int]
        :param kernel_valuess: [float], contains the default values of the parameters of the kernel
        :param mean_values: [[float]], It contains the value of the mean parameter.
        :param var_noise_values: [[float]], It contains the variance of the noise of the model

        :return: dict
        """
        if optimize_only_posterior_means is None:
            optimize_only_posterior_means = [False]

        if start_optimize_posterior_means is None:
            start_optimize_posterior_means = [0]
        if caches is None:
            caches = [True]

        if debugs is None:
            debugs = [False]

        if same_correlations is None:
            same_correlations = [True]

        if monte_carlo_sbos is None:
            monte_carlo_sbos = [True]

        if n_samples_mcs is None:
            n_samples_mcs = [1]

        if n_restarts_mcs is None:
            n_restarts_mcs = [1]

        if factr_mcs is None:
            factr_mcs = [1e12]

        if maxiter_mcs is None:
            maxiter_mcs = [1000]

        if use_only_training_pointss is None:
            use_only_training_pointss = [True]

        if n_restartss is None:
            n_restartss = [10]

        if n_best_restartss is None:
            n_best_restartss = [0]

        if n_best_restarts_mcs is None:
            n_best_restarts_mcs = [1]

        if n_samples_parameterss is None:
            n_samples_parameterss = [5]

        if n_restarts_means is None:
            n_restarts_means = [1000]

        if n_best_restarts_means is None:
            n_best_restarts_means = [100]

        if method_opt_mcs is None:
            method_opt_mcs = [DOGLEG]

        if maxepochs is None:
            method_opt_mcs = [10]

        if n_samples_parameters_means is None:
            n_samples_parameters_means = [15]

        if maxepoch_means is None:
            maxepoch_means = [20]

        if threshold_sbos is None:
            threshold_sbos = [None]


        if kernel_valuess is None:
            kernel_valuess = [[]]

        if mean_values is None:
            mean_values = [[]]

        if var_noise_values is None:
            var_noise_values = [[]]

        if name_models is None:
            name_models = ['gp_fitting_gaussian']

        if parallel_trainings is None:
            parallel_trainings = [True]

        if minimizes is None:
            minimizes = [False]

        if n_iterationss is None:
            n_iterationss = [5]

        if x_domains is None:
            x_domains = [[]]

        if parameters_distributions is None:
            parameters_distributions = [{}]

        if distributions is None:
            distributions = [UNIFORM_FINITE]

        if len(problem_names) != n_specs:
            problem_names = n_specs * problem_names

        if len(dim_xs) != n_specs:
            dim_xs = n_specs * dim_xs

        if len(start_optimize_posterior_means) != n_specs:
            start_optimize_posterior_means = n_specs * start_optimize_posterior_means

        if len(bounds_domain_xs) != n_specs:
            bounds_domain_xs = n_specs * bounds_domain_xs

        if mles is None:
            mles = [True]

        if thinnings is None:
            thinnings = [0]

        if n_burnings is None:
            n_burnings = [0]

        if max_steps_outs is None:
            max_steps_outs = [1]

        if training_datas is None:
            training_datas = [{}]

        if choose_noises is None:
            choose_noises = [True]

        if method_optimizations is None:
            method_optimizations = ['sbo']

        if n_trainings is None:
            n_trainings = [10]

        if noises is None:
            noises = [False]

        if n_sampless is None:
            n_sampless = [0]

        if parallels is None:
            parallels = [True]

        if random_seeds is None:
            random_seeds = [DEFAULT_RANDOM_SEED]

        if bounds_domains is None:
            bounds_domains = []
            for bounds_domain_x in bounds_domain_xs:
                bounds_domains.append([[bound[0], bound[1]] for bound in bounds_domain_x])

        if len(bounds_domains) != n_specs:
            bounds_domains = n_specs * bounds_domains

        if number_points_each_dimensions is None:
            number_points_each_dimensions = []
            for dim_x in dim_xs:
                number_points_each_dimensions.append(dim_x * [10])

        if len(optimize_only_posterior_means) != n_specs:
            optimize_only_posterior_means = n_specs * optimize_only_posterior_means

        if len(number_points_each_dimensions) != n_specs:
            number_points_each_dimensions = n_specs * number_points_each_dimensions

        if number_points_each_dimension_debugs is None:
            number_points_each_dimension_debugs = []
            for dim_x in dim_xs:
                number_points_each_dimension_debugs.append(dim_x * [10])

        if len(number_points_each_dimension_debugs) != n_specs:
            number_points_each_dimension_debugs = n_specs * number_points_each_dimension_debugs

        if len(training_names) != n_specs:
            training_names = n_specs * training_names

        if len(choose_noises) != n_specs:
            choose_noises = n_specs * choose_noises

        if len(method_optimizations) != n_specs:
            method_optimizations = n_specs * method_optimizations

        if type_boundss is None:
            type_boundss = []
            for bounds_domain in bounds_domains:
                type_boundss.append(len(bounds_domain) * [0])

        if len(type_boundss) != n_specs:
            type_boundss = n_specs * type_boundss

        if len(n_trainings) != n_specs:
            n_trainings = n_specs * n_trainings

        if pointss is None:
            pointss = n_specs * [[]]

        if len(noises) != n_specs:
            noises = n_specs * noises

        if len(n_sampless) != n_specs:
            n_sampless = n_specs * n_sampless

        if len(random_seeds) != n_specs:
            random_seeds = n_specs * random_seeds

        if len(parallels) != n_specs:
            parallels = n_specs * parallels

        if len(type_kernels) != n_specs:
            type_kernels = n_specs * type_kernels

        if len(dimensionss) != n_specs:
            dimensionss = n_specs * dimensionss

        if len(name_models) != n_specs:
            name_models = n_specs * name_models

        if len(mles) != n_specs:
            mles = n_specs * mles

        if len(thinnings) != n_specs:
            thinnings = n_specs * thinnings

        if len(n_burnings) != n_specs:
            n_burnings = n_specs * n_burnings

        if len(parallel_trainings) != n_specs:
            parallel_trainings = n_specs * parallel_trainings

        if len(max_steps_outs) != n_specs:
            max_steps_outs = n_specs * max_steps_outs

        if len(training_datas) != n_specs:
            training_datas = n_specs * training_datas

        if len(x_domains) != n_specs:
            x_domains = n_specs * x_domains

        if len(distributions) != n_specs:
            distributions = n_specs * distributions

        if len(parameters_distributions) != n_specs:
            parameters_distributions = n_specs * parameters_distributions

        if len(minimizes) != n_specs:
            minimizes = n_specs * minimizes

        if len(n_iterationss) != n_specs:
            n_iterationss = n_specs * n_iterationss

        if len(kernel_valuess) != n_specs:
            kernel_valuess = n_specs * kernel_valuess

        if len(mean_values) != n_specs:
            mean_values = n_specs * mean_values

        if len(var_noise_values) != n_specs:
            var_noise_values = n_specs * var_noise_values

        if len(caches) != n_specs:
            caches = n_specs * caches

        if len(debugs) != n_specs:
            debugs = n_specs * debugs

        if len(same_correlations) != n_specs:
            same_correlations = n_specs * same_correlations

        if len(monte_carlo_sbos) != n_specs:
            monte_carlo_sbos = n_specs * monte_carlo_sbos

        if len(n_samples_mcs) != n_specs:
            n_samples_mcs = n_specs * n_samples_mcs

        if len(n_restarts_mcs) != n_specs:
            n_restarts_mcs = n_specs * n_restarts_mcs

        if len(factr_mcs) != n_specs:
            factr_mcs = n_specs * factr_mcs

        if len(maxiter_mcs) != n_specs:
            maxiter_mcs = n_specs * maxiter_mcs

        if len(use_only_training_pointss) != n_specs:
            use_only_training_pointss = n_specs * use_only_training_pointss

        if len(n_restartss) != n_specs:
            n_restartss = n_specs * n_restartss

        if len(n_best_restartss) != n_specs:
            n_best_restartss = n_specs * n_best_restartss

        if len(n_best_restarts_mcs) != n_specs:
            n_best_restarts_mcs = n_specs * n_best_restarts_mcs

        if len(n_samples_parameterss) != n_specs:
            n_samples_parameterss = n_specs * n_samples_parameterss

        if len(n_restarts_means) != n_specs:
            n_restarts_means = n_specs * n_restarts_means

        if len(n_best_restarts_means) != n_specs:
            n_best_restarts_means = n_specs * n_best_restarts_means

        if len(method_opt_mcs) != n_specs:
            method_opt_mcs = n_specs * method_opt_mcs

        if len(maxepochs) != n_specs:
            maxepochs = n_specs * maxepochs

        if len(n_samples_parameters_means) != n_specs:
            n_samples_parameters_means = n_specs * n_samples_parameters_means

        if len(maxepoch_means) != n_specs:
            maxepoch_means = n_specs * maxepoch_means

        if len(threshold_sbos) != n_specs:
            threshold_sbos = n_specs * threshold_sbos

        return {
            'problem_names': problem_names,
            'dim_xs': dim_xs,
            'bounds_domain_xs': bounds_domain_xs,
            'training_names': training_names,
            'bounds_domains': bounds_domains,
            'number_points_each_dimensions': number_points_each_dimensions,
            'choose_noises': choose_noises,
            'method_optimizations': method_optimizations,
            'type_boundss': type_boundss,
            'n_trainings': n_trainings,
            'pointss': pointss,
            'noises': noises,
            'n_sampless': n_sampless,
            'random_seeds': random_seeds,
            'parallels': parallels,
            'type_kernels': type_kernels,
            'dimensionss': dimensionss,
            'name_models': name_models,
            'mles': mles,
            'thinnings': thinnings,
            'n_burnings': n_burnings,
            'max_steps_outs': max_steps_outs,
            'training_datas': training_datas,
            'x_domains': x_domains,
            'distributions': distributions,
            'parameters_distributions': parameters_distributions,
            'n_iterationss': n_iterationss,
            'minimizes': minimizes,
            'var_noise_values': var_noise_values,
            'mean_values': mean_values,
            'kernel_valuess': kernel_valuess,
            'caches': caches,
            'debugs': debugs,
            'same_correlations': same_correlations,
            'number_points_each_dimension_debugs': number_points_each_dimension_debugs,
            'monte_carlo_sbos': monte_carlo_sbos,
            'n_samples_mcs': n_samples_mcs,
            'n_restarts_mcs': n_restarts_mcs,
            'factr_mcs': factr_mcs,
            'maxiter_mcs': maxiter_mcs,
            'use_only_training_pointss': use_only_training_pointss,
            'n_restartss': n_restartss,
            'n_best_restartss': n_best_restartss,
            'n_best_restarts_mcs': n_best_restarts_mcs,
            'n_samples_parameterss': n_samples_parameterss,
            'n_restarts_means': n_restarts_means,
            'n_best_restarts_means': n_best_restarts_means,
            'method_opt_mcs': method_opt_mcs,
            'maxepochs': maxepochs,
            'n_samples_parameters_means': n_samples_parameters_means,
            'maxepoch_means': maxepoch_means,
            'threshold_sbos': threshold_sbos,
            'parallel_trainings': parallel_trainings,
            'optimize_only_posterior_means': optimize_only_posterior_means,
            'start_optimize_posterior_means': start_optimize_posterior_means,
        }

    @classmethod
    def generate_specs(cls, n_spec, multiple_spec):
        """
        Generate the spec associated to the ith entries of multiple_spec.

        :param n_spec: (int) Number of specificaction to be created
        :param multiple_spec: MultipleSpecEntity

        :return: RunSpecEntity
        """

        dim_xs = multiple_spec.get('dim_xs')[n_spec]
        problem_names = multiple_spec.get('problem_names')[n_spec]
        method_optimizations = multiple_spec.get('method_optimizations')[n_spec]
        choose_noises = multiple_spec.get('choose_noises')[n_spec]
        bounds_domain_xs = multiple_spec.get('bounds_domain_xs')[n_spec]
        number_points_each_dimensions = multiple_spec.get('number_points_each_dimensions')[n_spec]

        training_names = multiple_spec.get('training_names')[n_spec]
        bounds_domains = multiple_spec.get('bounds_domains')[n_spec]
        type_boundss = multiple_spec.get('type_boundss')[n_spec]
        n_trainings = multiple_spec.get('n_trainings')[n_spec]
        pointss = multiple_spec.get('pointss')[n_spec]
        noises = multiple_spec.get('noises')[n_spec]
        n_sampless = multiple_spec.get('n_sampless')[n_spec]
        random_seeds = multiple_spec.get('random_seeds')[n_spec]
        parallels = multiple_spec.get('parallels')[n_spec]

        # New parameters due to the GP model
        name_models = multiple_spec.get('name_models')[n_spec]
        type_kernels = multiple_spec.get('type_kernels')[n_spec]
        dimensionss = multiple_spec.get('dimensionss')[n_spec]
        mles = multiple_spec.get('mles')[n_spec]
        thinnings = multiple_spec.get('thinnings')[n_spec]
        n_burnings = multiple_spec.get('n_burnings')[n_spec]
        max_steps_outs = multiple_spec.get('max_steps_outs')[n_spec]
        training_datas = multiple_spec.get('training_datas')[n_spec]

        # New parameters due Bayesian quadrature
        x_domains = multiple_spec.get('x_domains')[n_spec]
        distributions = multiple_spec.get('distributions')[n_spec]

        parameters_distributions = multiple_spec.get('parameters_distributions')[n_spec]

        minimizes = multiple_spec.get('minimizes')[n_spec]
        n_iterationss = multiple_spec.get('n_iterationss')[n_spec]

        kernel_valuess = multiple_spec.get('kernel_valuess')[n_spec]
        mean_values = multiple_spec.get('mean_values')[n_spec]
        var_noise_values = multiple_spec.get('var_noise_values')[n_spec]

        caches = multiple_spec.get('caches')[n_spec]
        debugs = multiple_spec.get('debugs')[n_spec]

        same_correlations = multiple_spec.get('same_correlations')[n_spec]

        number_points_each_dimension_debugs = \
            multiple_spec.get('number_points_each_dimension_debugs')[n_spec]

        monte_carlo_sbos = multiple_spec.get('monte_carlo_sbos')[n_spec]
        n_samples_mcs = multiple_spec.get('n_samples_mcs')[n_spec]
        n_restarts_mcs = multiple_spec.get('n_restarts_mcs')[n_spec]

        factr_mcs = multiple_spec.get('factr_mcs')[n_spec]
        maxiter_mcs = multiple_spec.get('maxiter_mcs')[n_spec]

        use_only_training_pointss = multiple_spec.get('use_only_training_pointss')[n_spec]
        n_restartss = multiple_spec.get('n_restartss')[n_spec]
        n_best_restartss = multiple_spec.get('n_best_restartss')[n_spec]

        n_best_restarts_mcs = multiple_spec.get('n_best_restarts_mcs')[n_spec]

        n_samples_parameterss = multiple_spec.get('n_samples_parameterss')[n_spec]

        n_restarts_means = multiple_spec.get('n_restarts_means')[n_spec]
        n_best_restarts_means = multiple_spec.get('n_best_restarts_means')[n_spec]

        method_opt_mcs = multiple_spec.get('method_opt_mcs')[n_spec]
        maxepochs = multiple_spec.get('maxepochs')[n_spec]

        n_samples_parameters_means = multiple_spec.get('n_samples_parameters_means')[n_spec]

        maxepoch_means = multiple_spec.get('maxepoch_means')[n_spec]

        threshold_sbos = multiple_spec.get('threshold_sbos')[n_spec]

        parallel_trainings = multiple_spec.get('parallel_trainings')[n_spec]

        optimize_only_posterior_mean = multiple_spec.get('optimize_only_posterior_means')[n_spec]
        start_optimize_posterior_mean = multiple_spec.get('start_optimize_posterior_means')[n_spec]

        entry = {}

        entry.update({
            'problem_name': problem_names,
            'dim_x': dim_xs,
            'optimize_only_posterior_mean': optimize_only_posterior_mean,
            'start_optimize_posterior_mean': start_optimize_posterior_mean,
            'choose_noise': choose_noises,
            'bounds_domain_x': bounds_domain_xs,
            'number_points_each_dimension': number_points_each_dimensions,
            'method_optimization': method_optimizations,
            'training_name': training_names,
            'bounds_domain': bounds_domains,
            'n_training': n_trainings,
            'points': pointss,
            'noise': noises,
            'n_samples': n_sampless,
            'random_seed': random_seeds,
            'parallel': parallels,
            'type_bounds': type_boundss,
            'name_model': name_models,
            'type_kernel': type_kernels,
            'dimensions': dimensionss,
            'mle': mles,
            'thinning': thinnings,
            'n_burning': n_burnings,
            'max_steps_out': max_steps_outs,
            'training_data': training_datas,
            'x_domain': x_domains,
            'distribution': distributions,
            'parameters_distribution': parameters_distributions,
            'minimize': minimizes,
            'n_iterations': n_iterationss,
            'kernel_values': kernel_valuess,
            'mean_value': mean_values,
            'var_noise_value': var_noise_values,
            'cache': caches,
            'debug': debugs,
            'same_correlation': same_correlations,
            'number_points_each_dimension_debug': number_points_each_dimension_debugs,
            'monte_carlo_sbo': monte_carlo_sbos,
            'n_samples_mc': n_samples_mcs,
            'n_restarts_mc': n_restarts_mcs,
            'factr_mc': factr_mcs,
            'maxiter_mc': maxiter_mcs,
            'use_only_training_points': use_only_training_pointss,
            'n_restarts': n_restartss,
            'n_best_restarts_mc': n_best_restarts_mcs,
            'n_best_restarts': n_best_restartss,
            'n_samples_parameters': n_samples_parameterss,
            'n_best_restarts_mean': n_best_restarts_means,
            'n_restarts_mean': n_restarts_means,
            'method_opt_mc': method_opt_mcs,
            'maxepoch': maxepochs,
            'n_samples_parameters_mean': n_samples_parameters_means,
            'maxepoch_mean': maxepoch_means,
            'threshold_sbo': threshold_sbos,
            'parallel_training': parallel_trainings,
        })

        run_spec = RunSpecEntity(entry)

        return run_spec

        # problem_names = multiple_spec.problem_names
        # method_optimizations = multiple_spec.method_optimizations
        # dim_xs = multiple_spec.dim_xs
        # choose_noises = multiple_spec.choose_noises
        # bounds_domain_xs = multiple_spec.bounds_domain_xs
        # number_points_each_dimensions = multiple_spec.number_points_each_dimensions
        #
        # training_names = multiple_spec.training_names
        # bounds_domains = multiple_spec.bounds_domains
        # type_boundss = multiple_spec.type_boundss
        # n_trainings = multiple_spec.n_trainings
        #
        # type_kernels = multiple_spec.type_kernels
        # dimensionss = multiple_spec.dimensionss
        #
        # name_models = multiple_spec.name_models
        # if name_models is None:
        #     name_models = n_specs * ['gp_fitting_gaussian']
        #
        # mles = multiple_spec.mles
        # if mles is None:
        #     mles = n_specs * [True]
        #
        # thinnings = multiple_spec.thinnings
        # if thinnings is None:
        #     thinnings = n_specs * [0]
        #
        # n_burnings = multiple_spec.n_burnings
        # if n_burnings is None:
        #     n_burnings = n_specs * [0]
        #
        # max_steps_outs = multiple_spec.max_steps_outs
        # if max_steps_outs is None:
        #     max_steps_outs = n_specs * [1]
        #
        # training_datas = multiple_spec.training_datas
        # if training_datas is None:
        #     training_datas = n_specs * [{}]
        #
        # pointss = multiple_spec.pointss
        # if pointss is None:
        #     pointss = n_specs * [[]]
        #
        # noises = multiple_spec.noises
        # if noises is None:
        #     noises = n_specs * [False]
        #
        # n_sampless = multiple_spec.n_sampless
        # if n_sampless is None:
        #     n_sampless = n_specs * [0]
        #
        # random_seeds = multiple_spec.random_seeds
        # if random_seeds is None:
        #     random_seeds = n_specs * [DEFAULT_RANDOM_SEED]
        #
        # parallels = multiple_spec.parallels
        # if parallels is None:
        #     parallels = n_specs * [True]
        #
        # x_domains = multiple_spec.x_domains
        # if x_domains is None:
        #     x_domains = n_specs * [[]]
        #
        # distributions = multiple_spec.distributions
        # if distributions is None:
        #     distributions = n_specs * [UNIFORM_FINITE]
        #
        # parameters_distributions = multiple_spec.parameters_distributions
        # if parameters_distributions is None:
        #     parameters_distributions = n_specs * [{}]
        #
        # minimizes = multiple_spec.minimizes
        # if minimizes is None:
        #     minimizes = n_specs * [False]
        #
        # n_iterationss = multiple_spec.n_iterationss
        # if n_iterationss is None:
        #     n_iterationss = n_specs * [5]
        #
        # kernel_valuess = multiple_spec.kernel_valuess
        # if kernel_valuess is None:
        #     kernel_valuess = n_specs * [[]]
        #
        # mean_values = multiple_spec.mean_values
        # if mean_values is None:
        #     mean_values = n_specs * [[]]
        #
        # var_noise_values = multiple_spec.var_noise_values
        # if var_noise_values is None:
        #     var_noise_values = n_specs * [[]]
        #
        # run_spec = []
        #
        # for problem_name, method_optimization, dim_x, choose_noise, bounds_domain_x, \
        #     number_points_each_dimension, training_name, bounds_domain, type_bounds, n_training, \
        #     points, noise, n_samples, random_seed, parallel, type_kernel, dimensions, name_model, \
        #     mle, thinning, n_burning, max_steps_out, training_data, x_domain, distribution, \
        #     parameters_distribution, minimize, n_iterations, kernel_values, mean_value, \
        #     var_noise_value  in \
        #         zip(problem_names, method_optimizations, dim_xs, choose_noises, bounds_domain_xs,
        #             number_points_each_dimensions, training_names, bounds_domains, type_boundss,
        #             n_trainings, pointss, noises, n_sampless, random_seeds, parallels, type_kernels,
        #             dimensionss, name_models, mles, thinnings, n_burnings, max_steps_outs,
        #             training_datas, x_domains, distributions, parameters_distributions, minimizes,
        #             n_iterationss, kernel_valuess, mean_values, var_noise_values):
        #
        #     parameters_entity = {
        #         'problem_name': problem_name,
        #         'method_optimization': method_optimization,
        #         'dim_x': dim_x,
        #         'choose_noise': choose_noise,
        #         'bounds_domain_x': bounds_domain_x,
        #         'number_points_each_dimension': number_points_each_dimension,
        #         'training_name': training_name,
        #         'bounds_domain': bounds_domain,
        #         'type_bounds': type_bounds,
        #         'n_training': n_training,
        #         'points': points,
        #         'noise': noise,
        #         'n_samples': n_samples,
        #         'random_seed': random_seed,
        #         'parallel': parallel,
        #         'type_kernel': type_kernel,
        #         'dimensions': dimensions,
        #         'name_model': name_model,
        #         'mle': mle,
        #         'thinning': thinning,
        #         'n_burning': n_burning,
        #         'max_steps_out': max_steps_out,
        #         'training_data': training_data,
        #         'x_domain': x_domain,
        #         'distribution': distribution,
        #         'parameters_distribution': parameters_distribution,
        #         'minimize': minimize,
        #         'n_iterations': n_iterations,
        #         'var_noise_value': var_noise_value,
        #         'mean_value': mean_value,
        #         'kernel_values': kernel_values,
        #     }
        #
        #     run_spec.append(RunSpecEntity(parameters_entity))

    @classmethod
    def collect_multi_spec_results(cls, multiple_spec, total_iterations=None, sign=True, sqr=False,
                                   same_random_seeds=False, rs_lw=0, rs_up=None,
                                   combine_method=None):
        """
        Writes the files with the aggregated results
        :param multiple_spec:
        :param total_iterations: (int) Collect results until this iteration
        :param sign: (boolean) If true, we multiply the results by -1
        :param sqr: (boolean) If true, we take the square root of the results
        :param same_random_seeds: (boolean) If true, we use the same random seeds for both problems
        :param combine_method: (str) Name of the method to combine its aggregate results with new
            runs. The aggregate results should be located in the path: aggregate_results/respaldo/
            It does not work with same_random_seeds
        :return:
        """

        if total_iterations is None:
            total_iterations = 10000

        n_specs = len(multiple_spec.get('random_seeds'))

        results_dict = {}

        if sign:
            sign = -1.0
        else:
            sign = 1.0

        if sqr:
            f = lambda x: x ** 0.5
        else:
            f = lambda x: x

        if rs_up is not None:
            same_random_seeds = True

        if same_random_seeds:
            random_seeds = {}
            for method in set(multiple_spec.get('method_optimizations')):
                random_seeds[method] = []
            for i in range(n_specs):
                problem_name = multiple_spec.get('problem_names')[i]
                dir = path.join(PROBLEM_DIR, problem_name, PARTIAL_RESULTS)

                if not os.path.exists(dir):
                    continue

                training_name = multiple_spec.get('training_names')[i]
                n_training = multiple_spec.get('n_trainings')[i]
                random_seed = multiple_spec.get('random_seeds')[i]
                method = multiple_spec.get('method_optimizations')[i]
                n_samples_parameters = multiple_spec.get('n_samples_parameterss')[i]
                n_iterations = multiple_spec.get('n_iterationss')[i]

                file_name = cls._filename_results(
                    problem_name=problem_name,
                    training_name=training_name,
                    n_points=n_training,
                    random_seed=random_seed,
                    method=method,
                    n_samples_parameters=n_samples_parameters,
                )

                file_path = path.join(dir, file_name)
                if not os.path.exists(file_path):
                    continue
                random_seeds[method].append(random_seed)

            methods = list(set(multiple_spec.get('method_optimizations')))
            random_seeds_check = set(random_seeds[methods[0]])
            for i in range(1, len(methods)):
                random_seeds_check = random_seeds_check.intersection(random_seeds[methods[i]])

            if rs_up is not None:
                random_seeds_check = random_seeds_check.intersection(range(rs_lw, rs_up))

        for i in xrange(n_specs):
            problem_name = multiple_spec.get('problem_names')[i]
            dir = path.join(PROBLEM_DIR, problem_name, PARTIAL_RESULTS)

            if not os.path.exists(dir):
                continue

            training_name = multiple_spec.get('training_names')[i]
            n_training = multiple_spec.get('n_trainings')[i]
            random_seed = multiple_spec.get('random_seeds')[i]
            method = multiple_spec.get('method_optimizations')[i]
            n_samples_parameters = multiple_spec.get('n_samples_parameterss')[i]
            n_iterations = multiple_spec.get('n_iterationss')[i]

            if same_random_seeds and random_seed not in random_seeds_check:
                continue

            file_name = cls._filename_results(
                problem_name=problem_name,
                training_name=training_name,
                n_points=n_training,
                random_seed=random_seed,
                method=method,
                n_samples_parameters=n_samples_parameters,
            )

            file_path = path.join(dir, file_name)

            if not os.path.exists(file_path):
                continue

            results = JSONFile.read(file_path)
            results = results['objective_values']

            key_dict = (problem_name, training_name, n_training, method)
            if key_dict not in results_dict:
                results_dict[key_dict] = \
                    [[] for _ in range(min(n_iterations + 1, total_iterations))]

            for iteration in range(min(total_iterations, n_iterations + 1, len(results))):
                results_dict[key_dict][iteration].append(f(sign * results[iteration]))

        problem_names = list(set(multiple_spec.get('problem_names')))
        training_names = set(multiple_spec.get('training_names'))
        n_trainings = set(multiple_spec.get('n_trainings'))
        methods = set(multiple_spec.get('method_optimizations'))

        aggregated_results = {}

        for problem in problem_names:
            for training in training_names:
                for n_training in n_trainings:
                    for method in methods:

                        key = (problem, training, n_training, method)
                        aggregated_results[key] = {}

                        if key not in results_dict:
                            continue

                        data_aggregate = None
                        if combine_method is not None and combine_method == method:
                            dir_aggregate = path.join(
                                PROBLEM_DIR, problem, AGGREGATED_RESULTS, 'respaldo')
                            file_name_aggregate = cls._aggregated_results(
                                problem_name=problem,
                                training_name=training,
                                n_points=n_training,
                                method=method,
                            )
                            file_path_aggregate = path.join(dir_aggregate, file_name_aggregate)
                            data_aggregate = JSONFile.read(file_path_aggregate)

                        results = results_dict[key]

                        for iteration in xrange(min(len(results), total_iterations)):
                            if len(results[iteration]) > 0:
                                values = results[iteration]

                                mean = np.mean(values)
                                std = np.std(values)
                                n_samples = len(results[iteration])

                                if data_aggregate is not None:
                                    aggregate_iteration = data_aggregate[str(iteration)]
                                    mean_aggregate = aggregate_iteration['mean']
                                    n_samples_ag = aggregate_iteration['n_samples']
                                    std_ag = aggregate_iteration['std']

                                    old_mean = mean

                                    n_old_samples = float(n_samples)

                                    n_samples += n_samples_ag

                                    n_samples_ag = float(n_samples_ag)
                                    mean = (n_old_samples * mean) + \
                                           (mean_aggregate * float(n_samples_ag))
                                    mean /= float(n_samples)

                                    std_old = n_old_samples * (std ** 2)
                                    std_ag = n_samples_ag * (std_ag ** 2)
                                    third_term = n_old_samples * ((old_mean - mean) ** 2)
                                    fourth_term = n_samples_ag * ((mean_aggregate - mean) ** 2)

                                    std = std_old + std_ag + third_term + fourth_term
                                    std = np.sqrt(std / float(n_samples))

                                ci_low = mean - 1.96 * std / np.sqrt(n_samples)
                                ci_up = mean + 1.96 * std / np.sqrt(n_samples)

                                aggregated_results[key][iteration] = {}
                                aggregated_results[key][iteration]['mean'] = mean
                                aggregated_results[key][iteration]['std'] = std
                                aggregated_results[key][iteration]['n_samples'] = n_samples
                                aggregated_results[key][iteration]['ci_low'] = ci_low
                                aggregated_results[key][iteration]['ci_up'] = ci_up
                            else:
                                break

                        if len(aggregated_results[key]) > 0:
                            dir = path.join(PROBLEM_DIR, problem, AGGREGATED_RESULTS)

                            if not os.path.exists(dir):
                                os.mkdir(dir)

                            file_name = cls._aggregated_results(
                                problem_name=problem,
                                training_name=training,
                                n_points=n_training,
                                method=method,
                            )

                            file_path = path.join(dir, file_name)
                            JSONFile.write(aggregated_results[key], file_path)
