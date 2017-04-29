import unittest

from stratified_bayesian_optimization.entities.run_spec import RunSpecEntity, MultipleSpecEntity


class TestRunSpecEntity(unittest.TestCase):

    def test_from_json(self):
        spec = RunSpecEntity.from_json('test_spec.json')
        spec.validate()


class TestMultipleSpecEntity(unittest.TestCase):

    def test_from_json(self):
        multiple_spec = MultipleSpecEntity.from_json('multiple_test_spec.json')
        multiple_spec.validate()
