import unittest

from mock import create_autospec
from doubles import expect

import numpy as np

from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO
from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity
from stratified_bayesian_optimization.entities.domain import BoundsEntity, DomainEntity
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.constant import (
    SCALED_KERNEL,
    MATERN52_NAME,
    PRODUCT_KERNELS_SEPARABLE,
    TASKS_KERNEL_NAME,
    UNIFORM_FINITE,
)


class TestBGOService(unittest.TestCase):

    def setUp(self):

        dict = {
            'problem_name': 'test_problem_with_tasks',
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [BoundsEntity({'lower_bound': 0, 'upper_bound': 100})],
            'number_points_each_dimension': [4],
            'method_optimization': 'sbo',
            'training_name': 'test_bgo',
            'bounds_domain': [[0, 100], [0, 1]],
            'n_training': 100,
            'type_kernel': [PRODUCT_KERNELS_SEPARABLE, MATERN52_NAME, TASKS_KERNEL_NAME],
            'noise': False,
            'random_seed': 5,
            'parallel': True,
            'type_bounds': [0, 1],
            'dimensions': [2, 1, 2],
            'name_model': 'gp_fitting_gaussian',
            'mle': True,
            'thinning': 0,
            'n_burning': 0,
            'max_steps_out': 1,
            'training_data': None,
            'x_domain': [0],
            'distribution': UNIFORM_FINITE,
            'parameters_distribution': None,
            'minimize': False,
            'n_iterations': 5,
        }


        self.spec = RunSpecEntity(dict)


        self.acquisition_function = None
        self.gp_model = None

        #self.bgo = BGO(self.acquisition_function, self.gp_model)

    def test_from_spec(self):
       # bgo = BGO.from_spec(self.spec)
        assert True
        # TODO: FINISH THIS TEST

    def test_run_spec(self):
        bgo = create_autospec(BGO)
        expect(BGO).from_spec.and_return(bgo)
        domain = create_autospec(DomainEntity)
        expect(DomainService).from_dict.and_return(domain)
        expect(bgo).optimize.and_return({})

        assert BGO.run_spec(self.spec) == {}

    def test_optimize(self):
        bgo = BGO.from_spec(self.spec)
        z = bgo.optimize()
        assert z['optimal_solution'] == np.array([100.0])

