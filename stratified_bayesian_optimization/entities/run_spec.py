from __future__ import absolute_import

import ujson
from os import path

from schematics.models import Model
from schematics.types import IntType, StringType, BooleanType
from schematics.types.compound import ModelType, ListType

from stratified_bayesian_optimization.lib.constant import SPECS_DIR, MULTIPLESPECS_DIR
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
        choose_noise = spec['choose_noise']

        bounds_domain_x = spec['bounds_domain_x']
        bounds_domain_x = BoundsEntity.to_bounds_entity(bounds_domain_x)

        method_optimization = spec['method_optimization']
        problem_name = spec['problem_name']

        number_points_each_dimension = spec.get('number_points_each_dimension')

        entry.update({
            'problem_name': problem_name,
            'dim_x': dim_x,
            'choose_noise': choose_noise,
            'bounds_domain_x': bounds_domain_x,
            'number_points_each_dimension': number_points_each_dimension,
            'method_optimization': method_optimization
        })

        return cls(entry)


class MultipleSpecEntity(Model):
    dim_xs = ListType(IntType, required=True, min_size=1)
    problem_names = ListType(StringType, required=True, min_size=1)
    method_optimizations = ListType(StringType, required=True, min_size=1)
    choose_noises = ListType(BooleanType, required=True, min_size=1)
    bounds_domain_xs = ListType(ListType(ModelType(BoundsEntity), min_size=1, required=True),
                                required=True, min_size=1)
    number_points_each_dimensions = ListType(ListType(IntType), min_size=1, required=True)

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

        entry.update({
            'problem_names': problem_names,
            'dim_xs': dim_xs,
            'choose_noises': choose_noises,
            'bounds_domain_xs': bounds_domain_xs,
            'number_points_each_dimensions': number_points_each_dimensions,
            'method_optimizations': method_optimizations
        })

        return cls(entry)
