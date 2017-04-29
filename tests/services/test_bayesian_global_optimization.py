import unittest

from mock import create_autospec
from doubles import expect

from stratified_bayesian_optimization.services.bayesian_global_optimization import BGO
from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity
from stratified_bayesian_optimization.entities.domain import BoundsEntity, DomainEntity


class TestBGOService(unittest.TestCase):

    def setUp(self):
        self.spec = RunSpecEntity({
            'problem_name': 'toy',
            'method_optimization': 'SBO',
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [BoundsEntity({'lower_bound': 2, 'upper_bound': 3})],
            'number_points_each_dimension': [4]
        })

        self.acquisition_function = None
        self.gp_model = None

        self.bgo = BGO(self.acquisition_function, self.gp_model)

    def test_from_spec(self):
        BGO.from_spec(self.spec)

    def test_run_spec(self):
        bgo = create_autospec(BGO)
        expect(BGO).from_spec.and_return(bgo)
        domain = create_autospec(DomainEntity)
        expect(DomainEntity).from_dict.and_return(domain)
        expect(bgo).optimize.and_return({})

        assert BGO.run_spec(self.spec) == {}

    def test_optimize(self):
        assert {} == self.bgo.optimize(0, 1)
