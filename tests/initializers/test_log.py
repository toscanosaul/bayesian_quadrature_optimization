import unittest

from stratified_bayesian_optimization.initializers.log import SBOLog


class TestSBOLog(unittest.TestCase):

    def test_info(self):
        logger = SBOLog(__name__)
        logger.info('testing')
