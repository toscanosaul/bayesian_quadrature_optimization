from __future__ import absolute_import

from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity

logger = SBOLog(__name__)


class SpecService(object):
    @classmethod
    def generate_dict_spec(cls, problem_name, dim_x, choose_noise, bounds_domain_x,
                           number_points_each_dimension, method_optimization):
        """
        Generate dict that represents run spec.

        :param problem_name: (str)
        :param dim_x: int
        :param choose_noise: boolean
        :param bounds_domain_x: [(float, float)]
        :param number_points_each_dimension: [int] number of points in each dimension for the
            discretization of the domain of x.
        :param method_optimization: (str) Options: 'SBO', 'KG'
        :return: dict
        """

        return {
            'problem_name': problem_name,
            'dim_x': dim_x,
            'choose_noise': choose_noise,
            'bounds_domain_x': bounds_domain_x,
            'number_points_each_dimension': number_points_each_dimension,
            'method_optimization': method_optimization
        }

    # TODO - generate a list of runspecentities over different parameters

    @classmethod
    def generate_dict_multiple_spec(cls, problem_names, dim_xs, choose_noises, bounds_domain_xs,
                                    number_points_each_dimensions, method_optimizations):
        """
        Generate dict that represents multiple run specs

        :param problem_names: [str]
        :param dim_xs: [int]
        :param choose_noises: [boolean]
        :param bounds_domain_xs: [[(float, float)]]
        :param number_points_each_dimensions: [[int]]
        :param method_optimizations: [str]
        :return: dict
        """

        return {
            'problem_names': problem_names,
            'dim_xs': dim_xs,
            'choose_noises': choose_noises,
            'bounds_domain_xs': bounds_domain_xs,
            'number_points_each_dimensions': number_points_each_dimensions,
            'method_optimizations': method_optimizations
        }

    @classmethod
    def generate_specs(cls, multiple_spec):
        """
        Generate a list of RunSpecEntities.

        :param multiple_spec: MultipleSpecEntity

        :return: [RunSpecEntity]
        """

        problem_names = multiple_spec.problem_names
        method_optimizations = multiple_spec.method_optimizations
        dim_xs = multiple_spec.dim_xs
        choose_noises = multiple_spec.choose_noises
        bounds_domain_xs = multiple_spec.bounds_domain_xs
        number_points_each_dimensions = multiple_spec.number_points_each_dimensions

        run_spec = []

        for problem_name, method_optimization, dim_x, bounds_domain_x, choose_noise, \
            number_points_each_dimension in zip(problem_names, method_optimizations, dim_xs,
                                                bounds_domain_xs, choose_noises,
                                                number_points_each_dimensions):
            parameters_entity = {
                'problem_name': problem_name,
                'dim_x': dim_x,
                'choose_noise': choose_noise,
                'bounds_domain_x': bounds_domain_x,
                'number_points_each_dimension': number_points_each_dimension,
                'method_optimization': method_optimization
            }

            run_spec.append(RunSpecEntity(parameters_entity))

        return run_spec
