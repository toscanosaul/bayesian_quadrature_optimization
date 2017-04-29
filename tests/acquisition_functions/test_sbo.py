import unittest

from mock import create_autospec

from stratified_bayesian_optimization.acquisition_functions.sbo import SBO
from stratified_bayesian_optimization.entities.domain import DomainEntity


class TestSBO(unittest.TestCase):
    def setUp(self):
        self.kernel = None
        self.domain = create_autospec(DomainEntity)
        self.sbo = SBO(self.kernel, self.domain)

    def test_builder(self):
        SBO(self.kernel, self.domain)
