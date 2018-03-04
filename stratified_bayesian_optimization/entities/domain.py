from __future__ import absolute_import

import numpy as np
import itertools

from schematics.exceptions import ModelValidationError
from schematics.models import Model
from schematics.types import(
    IntType,
    FloatType,
    BooleanType,
)
from schematics.types.compound import ListType, ModelType

from stratified_bayesian_optimization.initializers.log import SBOLog


logger = SBOLog(__name__)


class BoundsEntity(Model):
    lower_bound = FloatType(required=True)
    upper_bound = FloatType(required=True)

    @staticmethod
    def get_bounds_as_lists(list_bounds):
        """
        Get a list with the bounds as lists.
        :param list_bounds: [BoundsEntity]
        :return: [[float, float]]
        """

        bounds = []
        for bound in list_bounds:
            bounds.append([bound.lower_bound, bound.upper_bound])
        return bounds

    @classmethod
    def to_bounds_entity(cls, list_list_bounds):
        """
        Get a list of bounds from a list of of lists of floats.
        :param list_list_bounds: [[float]]
        :return: [BoundsEntity]
        """

        list_bounds_entity = []

        for bound in list_list_bounds:
            list_bounds_entity.append(cls({'lower_bound': bound[0], 'upper_bound': bound[1]}))

        return list_bounds_entity

    def validate(self, *args, **kwargs):
        super(BoundsEntity, self).validate(*args, **kwargs)
        if self.lower_bound > self.upper_bound:
            raise ModelValidationError("Lower bound is greater than upper bound")


class DomainEntity(Model):
    dim_x = IntType(required=True)
    choose_noise = BooleanType(required=True)
    bounds_domain_x = ListType(ModelType(BoundsEntity), min_size=1, required=True)

    # Required only for SBO
    dim_w = IntType()
    bounds_domain_w = ListType(ModelType(BoundsEntity))
    domain_w = ListType(ListType(FloatType))
    discretization_domain_x = ListType(ListType(FloatType))

    @staticmethod
    def discretize_domain(bounds_domain, number_points_each_dimension):
        """
        Discretize uniformly a domain defined by bounds_domain.
        :param bounds_domain: ([BoundsEntity]) Each entry of the list contains the bounds of each
            dimension of the domain.
        :param number_points_each_dimension: ([int])
        :return: [[float]]
        """
        if len(number_points_each_dimension) != len(bounds_domain):
            raise ValueError("Dimensions are wrong!")

        points = []
        for bound, number_points in zip(bounds_domain, number_points_each_dimension):
            points.append(np.linspace(bound.lower_bound, bound.upper_bound, number_points))

        domain = []
        for point in itertools.product(*points):
            domain.append(list(point))

        return domain

    @staticmethod
    def check_dimension_each_entry(list_elements, dimension):
        """
        Check if each element of list_elements has dimension equal to dimension
        :param list_elements: ([[float]])
        :param dimension: (int)
        :return: boolean
        """
        for element in list_elements:
            if len(element) != dimension:
                return False
        return True

    # def validate(self, *args, **kwargs):
    #     super(DomainEntity, self).validate(*args, **kwargs)
    #     if len(self.bounds_domain_x) != self.dim_x:
    #         raise ModelValidationError("Wrong dimension of bounds_domain_x")
    #     if self.choose_noise and self.dim_w is None:
    #         raise ModelValidationError("Missing dimension of w")
    #     if self.choose_noise and self.domain_w is None and self.bounds_domain_w is None:
    #         raise ModelValidationError("Missing range of w")
    #     if self.choose_noise and self.discretization_domain_x is None:
    #         raise ModelValidationError("Missing discretization of domain of x")
    #     if self.choose_noise:
    #         correct_dim_x = self.check_dimension_each_entry(self.discretization_domain_x,
    #                                                         self.dim_x)
    #         if correct_dim_x is False:
    #             raise ModelValidationError("Wrong dimensions")
    #     if self.choose_noise and self.bounds_domain_w is not None:
    #         if len(self.bounds_domain_w) != self.dim_w:
    #             raise ModelValidationError("Wrong dimensions of the bounds of w")
    #     if self.choose_noise and self.domain_w is not None:
    #         correct_dim_w = self.check_dimension_each_entry(self.domain_w, self.dim_w)
    #         if correct_dim_w is False:
    #             raise ModelValidationError("Wrong dimensions")
