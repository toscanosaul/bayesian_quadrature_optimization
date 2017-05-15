from __future__ import absolute_import

import ujson
from os import path

from schematics.models import Model
from schematics.types import IntType, StringType, BooleanType, FloatType
from schematics.types.compound import ModelType, ListType

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
    choose_noise = BooleanType(required=True)
    bounds_domain_x = ListType(ModelType(BoundsEntity), min_size=1, required=True)
    number_points_each_dimension = ListType(IntType)
    training_name = StringType(required=True)
    bounds_domain = ListType(ListType(FloatType), min_size=1, required=True)
    type_bounds = ListType(IntType, required=True)
    n_training = IntType(required=True)
    points = ListType(ListType(FloatType), required=False)
    noise = BooleanType(required=False)
    n_samples = IntType(required=False)
    random_seed = IntType(required=False)
    parallel = BooleanType(required=False)


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
    random_seeds = ListType(IntType, required=False)
    parallels = ListType(BooleanType, required=False)

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

        type_boundss = spec.get('type_bounds')
        if type_boundss is None:
            type_boundss = []
            for bounds_domain in bounds_domains:
                type_boundss.append(len(bounds_domain) * [0])


        pointss = spec.get('pointss', None)
        noises = spec.get('noises', n_specs * [False])
        n_sampless = spec.get('n_samples',  n_specs * [0])
        random_seeds = spec.get('random_seed', n_specs * [DEFAULT_RANDOM_SEED])
        parallels = spec.get('parallel', n_specs * [True])

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
        })

        return cls(entry)
