from __future__ import absolute_import

from doubles import allow, expect
import unittest

from stratified_bayesian_optimization.services.domain import DomainService
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.entities.domain import(
    BoundsEntity,
    DomainEntity,
)


class TestDomainService(unittest.TestCase):

    def setUp(self):

        self.bounds_domain_x = BoundsEntity({
            'lower_bound': 0,
            'upper_bound': 1
        })

        self.spec = {
            'dim_x': 1,
            'choose_noise': True,
            'bounds_domain_x': [self.bounds_domain_x]
        }

    def test_load_discretization_file_not_exists(self):
        allow(JSONFile).read
        expect(JSONFile).write.once()
        expect(DomainEntity).discretize_domain.once().and_return([])
        expect(BoundsEntity).get_bounds_as_lists.once().and_return([2])

        assert DomainService.load_discretization(2, 1, 0) == []

    def test_load_discretization_file_exists(self):
        allow(JSONFile).read.and_return([])
        expect(DomainEntity).discretize_domain.never()
        expect(BoundsEntity).get_bounds_as_lists.once().and_return([2])

        assert DomainService.load_discretization(2, 1, 0) == []

    def test_domain_from_dict(self):
        expect(DomainService).load_discretization.never()
        DomainService.from_dict(self.spec)

        expect(DomainService).load_discretization.once().and_return([])
        self.spec['number_points_each_dimension'] = [5]
        self.spec['problem_name'] = 'test'
        DomainService.from_dict(self.spec)