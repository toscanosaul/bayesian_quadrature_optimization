import unittest

from stratified_bayesian_optimization.services.spec import SpecService
from stratified_bayesian_optimization.entities.run_spec import MultipleSpecEntity
from stratified_bayesian_optimization.entities.domain import BoundsEntity
from stratified_bayesian_optimization.lib.constant import (
    MATERN52_NAME,
    SCALED_KERNEL,
)


class TestSpecService(unittest.TestCase):

    def setUp(self):
        self.problem_name = 'toy'
        self.dim_x = 1
        self.choose_noise = True
        self.bounds_domain_x = [[2, 3]]
        self.number_points_each_dimension = [10]
        self.method_optimization = 'SBO'

        self.problem_names = ['toy']
        self.dim_xs = [1]
        self.choose_noises = [True]
        self.bounds_domain_xs = [[[2, 3]]]
        self.number_points_each_dimensions = [[5]]
        self.method_optimizations = ['SBO']

        self.bound = BoundsEntity({'lower_bound': 2, 'upper_bound': 3})
        self.training_name = 'test_spec'
        self.dimensions = [1]
        self.type_kernel = [SCALED_KERNEL, MATERN52_NAME]

    def test_generate_dict_spec(self):
        spec = SpecService.generate_dict_spec(self.problem_name, self.dim_x, self.bounds_domain_x,
                                              self.training_name, self.type_kernel, self.dimensions)

        assert spec == {
            'problem_name': self.problem_name,
            'dim_x': self.dim_x,
            'bounds_domain_x': self.bounds_domain_x,
            'training_name': self.training_name,
            'bounds_domain': self.bounds_domain_x,
            'number_points_each_dimension': [10],
            'choose_noise': True,
            'method_optimization': 'SBO',
            'type_bounds': [0],
            'n_training': 10,
            'points': [],
            'noise': False,
            'n_samples': 0,
            'random_seed': 1,
            'parallel': True,
            'type_kernel': [SCALED_KERNEL, MATERN52_NAME],
            'dimensions': [1],
            'name_model': 'gp_fitting_gaussian',
            'mle': True,
            'thinning': 0,
            'n_burning': 0,
            'max_steps_out': 1,
            'training_data': [],
        }

    def test_generate_dic_specs(self):
        specs = SpecService.generate_dict_multiple_spec(1, self.problem_names, self.dim_xs,
                                                        self.bounds_domain_xs, [self.training_name],
                                                        [self.type_kernel], [self.dimensions])

        assert specs == {
            'problem_names': self.problem_names,
            'dim_xs': self.dim_xs,
            'bounds_domain_xs': self.bounds_domain_xs,
            'training_names': [self.training_name],
            'bounds_domains': self.bounds_domain_xs,
            'number_points_each_dimensions': [[10]],
            'choose_noises': [True],
            'method_optimizations': ['SBO'],
            'type_boundss': [[0]],
            'n_trainings': [10],
            'pointss': [[]],
            'noises': [False],
            'n_sampless': [0],
            'random_seeds': [1],
            'parallels': [True],
            'type_kernels': [[SCALED_KERNEL, MATERN52_NAME]],
            'dimensionss': [[1]],
            'name_models': ['gp_fitting_gaussian'],
            'mles': [True],
            'thinnings': [0],
            'n_burnings': [0],
            'max_steps_outs': [1],
            'training_datas': [[]],

        }

        specs_ = SpecService.generate_dict_multiple_spec(2, ['toy', 'toy_2'], [1, 2],
                                                         [[[2, 3]], [[5, 8]]],
                                                         [self.training_name, 'test_spec_2'],
                                                         [self.type_kernel], [self.dimensions])

        assert specs_ == {
            'problem_names': ['toy', 'toy_2'],
            'dim_xs': [1, 2],
            'bounds_domain_xs': [[[2, 3]], [[5, 8]]],
            'training_names': [self.training_name, 'test_spec_2'],
            'bounds_domains': [[[2, 3]], [[5, 8]]],
            'number_points_each_dimensions': [[10], [10, 10]],
            'choose_noises': [True, True],
            'method_optimizations': ['SBO', 'SBO'],
            'type_boundss': [[0], [0]],
            'n_trainings': [10, 10],
            'pointss': [[], []],
            'noises': [False, False],
            'n_sampless': [0, 0],
            'random_seeds': [1, 1],
            'parallels': [True, True],
            'type_kernels': [[SCALED_KERNEL, MATERN52_NAME], [SCALED_KERNEL, MATERN52_NAME]],
            'dimensionss': [[1], [1]],
            'name_models': ['gp_fitting_gaussian', 'gp_fitting_gaussian'],
            'mles': [True, True],
            'thinnings': [0, 0],
            'n_burnings': [0, 0],
            'max_steps_outs': [1, 1],
            'training_datas': [[], []],
        }

    def test_generate_specs(self):
        multiple_spec = MultipleSpecEntity({
            'problem_names': self.problem_names,
            'dim_xs': self.dim_xs,
            'choose_noises': self.choose_noises,
            'bounds_domain_xs': [[self.bound]],
            'number_points_each_dimensions': self.number_points_each_dimensions,
            'method_optimizations': self.method_optimizations,
            'training_names': [self.training_name],
            'bounds_domains': self.bounds_domain_xs,
            'type_boundss': [[0]],
            'n_trainings': [10],
            'type_kernels': [self.type_kernel],
            'dimensionss': [self.dimensions]
        })

        specs = SpecService.generate_specs(1, multiple_spec)
        assert len(specs) == 1
        for s in specs:
            s.validate()
