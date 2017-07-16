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
    number_points_each_dimension = ListType(IntType)
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

    minimizes = ListType(BooleanType)
    n_iterationss = ListType(IntType)

    kernel_valuess = ListType(ListType(FloatType))
    mean_values = ListType(ListType(FloatType))
    var_noise_values = ListType(ListType(FloatType))

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

        entry.update({
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
        })

        return cls(entry)
