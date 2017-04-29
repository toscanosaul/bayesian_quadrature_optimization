import unittest

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity
from stratified_bayesian_optimization.entities.domain import BoundsEntity


class TestSpecService(unittest.TestCase):

    def setUp(self):
        self.problem_name = 'toy'
        self.dim_x = 1
        self.choose_noise = True
        self.bounds_domain_x = [[2, 3]]
        self.number_points_each_dimension = [5]
        self.method_optimization = 'SBO'

        self.problem_names = ['toy']
        self.dim_xs = [1]
        self.choose_noises = [True]
        self.bounds_domain_xs = [[[2, 3]]]
        self.number_points_each_dimensions = [[5]]
        self.method_optimizations = ['SBO']

        self.bound = BoundsEntity({'lower_bound': 2, 'upper_bound': 3})

    def test_generate_dict_spec(self):
        spec = SpecService.generate_dict_spec(self.problem_name, self.dim_x, self.choose_noise,
                                              self.bounds_domain_x,
                                              self.number_points_each_dimension,
                                              self.method_optimization)

        assert spec == {
            'problem_name': self.problem_name,
            'dim_x': self.dim_x,
            'choose_noise': self.choose_noise,
            'bounds_domain_x': self.bounds_domain_x,
            'number_points_each_dimension': self.number_points_each_dimension,
            'method_optimization': self.method_optimization
        }

    def test_generate_dic_specs(self):
        specs = SpecService.generate_dict_multiple_spec(self.problem_names, self.dim_xs,
                                                        self.choose_noises, self.bounds_domain_xs,
                                                        self.number_points_each_dimensions,
                                                        self.method_optimizations)

        assert specs == {
            'problem_names': self.problem_names,
            'dim_xs': self.dim_xs,
            'choose_noises': self.choose_noises,
            'bounds_domain_xs': self.bounds_domain_xs,
            'number_points_each_dimensions': self.number_points_each_dimensions,
            'method_optimizations': self.method_optimizations,
        }

    def test_generate_specs(self):
        multiple_spec = MultipleSpecEntity({
            'problem_names': self.problem_names,
            'dim_xs': self.dim_xs,
            'choose_noises': self.choose_noises,
            'bounds_domain_xs': [[self.bound]],
            'number_points_each_dimensions': self.number_points_each_dimensions,
            'method_optimizations': self.method_optimizations,
        })

        specs = SpecService.generate_specs(multiple_spec)
        assert len(specs) == 1
        for s in specs:
            s.validate()
