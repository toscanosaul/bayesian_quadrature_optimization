from __future__ import absolute_import

import ujson
from os import path

from schematics.models import Model
from schematics.types import IntType, StringType, BooleanType, FloatType, BaseType
from schematics.types.compound import ModelType, ListType, DictType

from stratified_bayesian_optimization.lib.constant import (
    SPECS_DIR,
    MULTIPLESPECS_DIR,
    DEFAULT_RANDOM_SEED,
    LBFGS_NAME,
    DOGLEG,
)
from stratified_bayesian_optimization.entities.domain import (
    BoundsEntity,
)


class RunSpecEntity(Model):
    problem_name = StringType(required=True)
    method_optimization = StringType(required=True)
    dim_x = IntType(required=True)
    choose_noise = BooleanType(required=True) #I think that we should remove it
    bounds_domain_x = ListType(ModelType(BoundsEntity), min_size=1, required=True)
    number_points_each_dimension = ListType(IntType) # Only for x
    training_name = StringType(required=True)
    bounds_domain = ListType(ListType(FloatType), min_size=1, required=True)
    type_bounds = ListType(IntType, required=True)
    n_training = IntType(required=True)
    points = ListType(ListType(FloatType), required=False)
    noise = BooleanType(required=False)
    n_samples = IntType(required=False)
    random_seed = IntType(required=True)
    parallel = BooleanType(required=False)

    name_model = StringType(required=False)
    type_kernel = ListType(StringType, required=True)
    dimensions = ListType(IntType, required=True)
    mle = BooleanType(required=False)
    thinning = IntType(required=False)
    n_burning = IntType(required=False)
    max_steps_out = IntType(required=False)
    training_data = BaseType()

    x_domain = ListType(IntType, required=False)
    distribution = StringType(required=False)
    parameters_distribution = DictType(ListType(FloatType), required=False)

    minimize = BooleanType(required=True)
    n_iterations = IntType(required=True)

    kernel_values = ListType(FloatType)
    mean_value = ListType(FloatType)
    var_noise_value = ListType(FloatType)
    same_correlation = BooleanType(required=False) # Used only for the task kernel

    cache = BooleanType(required=False)
    debug = BooleanType(required=False)

    number_points_each_dimension_debug = ListType(IntType) #Used to debug.

    # Parameter to estimate sbo by MC
    monte_carlo_sbo = BooleanType(required=False)
    n_samples_mc = IntType(required=False)
    n_restarts_mc = IntType(required=False)
    factr_mc = FloatType(required=False)
    maxiter_mc = IntType(required=False)
    n_best_restarts_mc = IntType(required=False)

    #Acquistion function parameters
    n_restarts = IntType(required=False)
    n_best_restarts = IntType(required=False)


    # We use only training points when reading GP model from cache
    use_only_training_points = BooleanType(required=False)

    # Computes everything using samples of the parameters if n_samples_parameters > 0
    n_samples_parameters = IntType(required=False)

    n_restarts_mean = IntType(required=False)
    n_best_restarts_mean = IntType(required=False)

    method_opt_mc = StringType(required=False)
    maxepoch = IntType(required=False)

    n_samples_parameters_mean = IntType(required=False)
    maxepoch_mean = IntType(required=False)

    threshold_sbo = FloatType(required=False)

    @classmethod
    def from_json(cls, specfile):
        """

        :param specfile: (str)
        :return: RunSpecEntity
        """

        with open(path.join(SPECS_DIR, specfile)) as specfile:
            spec = ujson.load(specfile)
            return cls.from_dictionary(spec)

    @classmethod
    def from_dictionary(cls, spec):
        """
        Create from dict

        :param spec: dict
        :return: RunSpecEntity
        """
        entry = {}
        dim_x = int(spec['dim_x'])
        choose_noise = spec.get('choose_noise', True)

        bounds_domain_x = spec['bounds_domain_x']
        bounds_domain_x = BoundsEntity.to_bounds_entity(bounds_domain_x)

        method_optimization = spec.get('method_optimization', 'SBO')
        problem_name = spec['problem_name']

        number_points_each_dimension = spec.get('number_points_each_dimension')

        training_name = spec.get('training_name')
        bounds_domain = spec.get('bounds_domain')
        n_training = spec.get('n_training', 10)
        type_bounds = spec.get('type_bounds', len(bounds_domain) * [0])

        points = spec.get('points', None)
        noise = spec.get('noise', False)
        n_samples = spec.get('n_samples', 0)
        random_seed = spec.get('random_seed', DEFAULT_RANDOM_SEED)
        parallel = spec.get('parallel', True)

        name_model = spec.get('name_model', 'gp_fitting_gaussian')
        type_kernel = spec.get('type_kernel')
        dimensions = spec.get('dimensions')
        mle = spec.get('mle', True)
        thinning = spec.get('thinning', 0)
        n_burning = spec.get('n_burning', 0)
        max_steps_out = spec.get('max_steps_out', 1)
        training_data = spec.get('training_data')

        x_domain = spec.get('x_domain')
        distribution = spec.get('distribution')
        parameters_distribution = spec.get('parameters_distribution')

        minimize = spec.get('minimize', False)
        n_iterations = spec.get('n_iterations', 5)

        kernel_values = spec.get('kernel_values')
        mean_value = spec.get('mean_value')
        var_noise_value = spec.get('var_noise_value')

        cache = spec.get('cache', True)
        debug = spec.get('debug', False)

        same_correlation = spec.get('same_correlation', False)

        number_points_each_dimension_debug = spec.get('number_points_each_dimension_debug')

        monte_carlo_sbo = spec.get('monte_carlo_sbo', False)
        n_samples_mc = spec.get('n_samples_mc', 100)
        n_restarts_mc = spec.get('n_restarts_mc', 100)

        factr_mc = spec.get('factr_mc', 1e12)
        maxiter_mc = spec.get('maxiter_mc', 1000)

        use_only_training_points = spec.get('use_only_training_points', True)
        n_restarts = spec.get('n_restarts', 10)
        n_best_restarts = spec.get('n_best_restarts', 10)

        n_best_restarts_mc = spec.get('n_best_restarts_mc', 10)

        n_samples_parameters = spec.get('n_samples_parameters', 0)

        n_restarts_mean = spec.get('n_restarts_mean', 1000)
        n_best_restarts_mean = spec.get('n_best_restarts_mean', 100)

        method_opt_mc = spec.get('method_opt_mc', LBFGS_NAME)
        maxepoch = spec.get('maxepoch', 10)

        n_samples_parameters_mean = spec.get('n_samples_parameters_mean', 15)

        maxepoch_mean = spec.get('maxepoch_mean', 15)

        threshold_sbo = spec.get('threshold_sbo')

        entry.update({
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
            'name_model': name_model,
            'type_kernel': type_kernel,
            'dimensions': dimensions,
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
            'kernel_values': kernel_values,
            'mean_value': mean_value,
            'var_noise_value': var_noise_value,
            'cache': cache,
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
            'n_best_restarts_mc': n_best_restarts_mc,
            'n_best_restarts': n_best_restarts,
            'n_samples_parameters': n_samples_parameters,
            'n_best_restarts_mean': n_best_restarts_mean,
            'n_restarts_mean': n_restarts_mean,
            'method_opt_mc': method_opt_mc,
            'maxepoch': maxepoch,
            'n_samples_parameters_mean': n_samples_parameters_mean,
            'maxepoch_mean': maxepoch_mean,
            'threshold_sbo': threshold_sbo,
        })

        return cls(entry)


# TODO - Modify MultipleSpecEntity to include the new parameters
class MultipleSpecEntity(Model):
    dim_xs = ListType(IntType, required=True, min_size=1)
    problem_names = ListType(StringType, required=True, min_size=1)
    method_optimizations = ListType(StringType, required=True, min_size=1)
    choose_noises = ListType(BooleanType, required=True, min_size=1)
    bounds_domain_xs = ListType(ListType(ModelType(BoundsEntity), min_size=1, required=True),
                                required=True, min_size=1)
    number_points_each_dimensions = ListType(ListType(IntType), min_size=1, required=True)

    training_names = ListType(StringType, required=True, min_size=1)
    bounds_domains = ListType(ListType(ListType(FloatType)), min_size=1, required=True)
    type_boundss = ListType(ListType(IntType, min_size=1), min_size=1, required=True)
    n_trainings = ListType(IntType, min_size=1, required=True)
    pointss = ListType(ListType(ListType(FloatType)), required=False)
    noises = ListType(BooleanType, required=False)
    n_sampless = ListType(IntType, required=False)
    random_seeds = ListType(IntType, required=True)
    parallels = ListType(BooleanType, required=False)

    # New parameters due to the GP model
    name_models = ListType(StringType, required=False)
    type_kernels = ListType(ListType(StringType), required=True, min_size=1)
    dimensionss = ListType(ListType(IntType), required=True, min_size=1)
    mles = ListType(BooleanType, required=False)
    thinnings = ListType(IntType, required=False)
    n_burnings = ListType(IntType, required=False)
    max_steps_outs = ListType(IntType, required=False)
    training_datas = ListType(BaseType())

    # New parameters due Bayesian quadrature
    x_domains = ListType(ListType(IntType), required=False)
    distributions = ListType(StringType, required=False)
    parameters_distributions = ListType(ListType(FloatType), required=False)

    minimizes = ListType(BooleanType, required=False)
    n_iterationss = ListType(IntType, required=False)

    kernel_valuess = ListType(ListType(FloatType), required=False)
    mean_values = ListType(ListType(FloatType), required=False)
    var_noise_values = ListType(ListType(FloatType), required=False)

    caches = ListType(BooleanType, required=False)
    debugs = ListType(BooleanType, required=False)

    same_correlations = ListType(BooleanType, required=False)

    number_points_each_dimension_debugs = ListType(ListType(IntType), required=False)

    monte_carlo_sbos = ListType(BooleanType, required=False)
    n_samples_mcs = ListType(IntType, required=False)
    n_restarts_mcs = ListType(IntType, required=False)

    factr_mcs = ListType(FloatType, required=False)
    maxiter_mcs = ListType(IntType, required=False)

    use_only_training_pointss = ListType(BooleanType, required=False)
    n_restartss = ListType(IntType, required=False)
    n_best_restartss = ListType(IntType, required=False)

    n_best_restarts_mcs = ListType(IntType, required=False)

    n_samples_parameterss = ListType(IntType, required=False)

    n_restarts_means = ListType(IntType, required=False)
    n_best_restarts_means = ListType(IntType, required=False)

    method_opt_mcs = ListType(StringType, required=False)
    maxepochs = ListType(IntType, required=False)

    n_samples_parameters_means = ListType(IntType, required=False)

    maxepoch_means = ListType(IntType, required=False)

    threshold_sbos = ListType(FloatType, required=False)

    # TODO - Complete all the other needed params

    @classmethod
    def from_json(cls, specfile):
        """

        :param specfile:
        :return: MultipleSpecEntity
        """
        with open(path.join(MULTIPLESPECS_DIR, specfile)) as specfile:
            spec = ujson.load(specfile)
            return cls.from_dictionary(spec)

    @classmethod
    def from_dictionary(cls, spec):
        """

        :param spec: dict
        :return: MultipleSpecEntity
        """

        entry = {}
        dim_xs = spec['dim_xs']
        choose_noises = spec['choose_noises']

        bounds_domain_xs_list = spec['bounds_domain_xs']

        bounds_domain_xs = []
        for bound in bounds_domain_xs_list:
            bounds_domain_xs.append(BoundsEntity.to_bounds_entity(bound))

        method_optimizations = spec['method_optimizations']
        problem_names = spec['problem_names']

        number_points_each_dimensions = spec.get('number_points_each_dimensions')

        n_specs = len(dim_xs)

        training_names = spec.get('training_names')
        bounds_domains = spec.get('bounds_domains')
        n_trainings = spec.get('n_trainings', n_specs * [10])

        type_boundss = spec.get('type_boundss', [len(bd) * [0] for bd in bounds_domains])

        pointss = spec.get('pointss', None)
        noises = spec.get('noisess', n_specs * [False])
        n_sampless = spec.get('n_sampless',  n_specs * [0])
        random_seeds = spec.get('random_seeds', n_specs * [DEFAULT_RANDOM_SEED])
        parallels = spec.get('parallels', n_specs * [True])

        name_models = spec.get('name_models', n_specs * ['gp_fitting_gaussian'])
        type_kernels = spec.get('type_kernels')
        dimensionss = spec.get('dimensionss')
        mles = spec.get('mles', n_specs * [True])
        thinnings = spec.get('thinnings', n_specs * [0])
        n_burnings = spec.get('n_burnings', n_specs * [0])
        max_steps_outs = spec.get('max_steps_outs', n_specs * [1])
        training_datas = spec.get('training_datas')

        x_domains = spec.get('x_domains')
        distributions = spec.get('distributions')
        parameters_distributions = spec.get('parameters_distributions')

        minimizes = spec.get('minimizes', n_specs * [False])
        n_iterationss = spec.get('n_iterationss', n_specs * [5])

        kernel_valuess = spec.get('kernel_valuess')
        mean_values = spec.get('mean_values')
        var_noise_values = spec.get('var_noise_values')


        caches = spec.get('caches', n_specs * [True])
        debugs = spec.get('debugs',  n_specs * [False])

        same_correlations = spec.get('same_correlations', n_specs * [True])

        number_points_each_dimension_debugs = spec.get('number_points_each_dimension_debugs')

        monte_carlo_sbos = spec.get('monte_carlo_sbos', n_specs * [True])
        n_samples_mcs = spec.get('n_samples_mcs', n_specs * [5])
        n_restarts_mcs = spec.get('n_restarts_mcs', n_specs * [5])

        factr_mcs = spec.get('factr_mcs', n_specs * [1e12])
        maxiter_mcs = spec.get('maxiter_mcs', n_specs * [1000])

        use_only_training_pointss = spec.get('use_only_training_pointss', n_specs * [True])
        n_restartss = spec.get('n_restartss', n_specs * [10])
        n_best_restartss = spec.get('n_best_restartss', n_specs * [0])

        n_best_restarts_mcs = spec.get('n_best_restarts_mcs', n_specs * [0])

        n_samples_parameterss = spec.get('n_samples_parameterss', n_specs * [0])

        n_restarts_means = spec.get('n_restarts_means', n_specs * [100])
        n_best_restarts_means = spec.get('n_best_restarts_means', n_specs * [10])

        method_opt_mcs = spec.get('method_opt_mcs', n_specs * [DOGLEG])
        maxepochs = spec.get('maxepochs', n_specs * [10])

        n_samples_parameters_means = spec.get('n_samples_parameters_means', n_specs * [20])

        maxepoch_means = spec.get('maxepoch_means', n_specs * [20])

        threshold_sbos = spec.get('threshold_sbos', n_specs * [[]])


        entry.update({
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
            'problem_names': problem_names,
            'dim_xs': dim_xs,
            'choose_noises': choose_noises,
            'bounds_domain_xs': bounds_domain_xs,
            'number_points_each_dimensions': number_points_each_dimensions,
            'method_optimizations': method_optimizations,
            'training_names': training_names,
            'bounds_domains': bounds_domains,
            'n_trainings': n_trainings,
            'pointss': pointss,
            'noises': noises,
            'n_sampless': n_sampless,
            'random_seeds': random_seeds,
            'parallels': parallels,
            'type_boundss': type_boundss,
            'name_models': name_models,
            'type_kernels': type_kernels,
            'dimensionss': dimensionss,
            'mles': mles,
            'thinnings': thinnings,
            'n_burnings': n_burnings,
            'max_steps_outs': max_steps_outs,
            'training_datas': training_datas,
            'distributions': distributions,
            'x_domains': x_domains,
            'parameters_distributions': parameters_distributions,
            'n_iterationss': n_iterationss,
            'minimizes': minimizes,
            'kernel_valuess': kernel_valuess,
            'mean_values': mean_values,
            'var_noise_values': var_noise_values,
            'threshold_sbos': threshold_sbos,
        })

        return cls(entry)
