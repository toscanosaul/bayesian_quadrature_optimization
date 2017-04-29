import unittest

from mock import mock_open, patch, MagicMock

from stratified_bayesian_optimization.util.json_file import JSONFile


class TestJSONFile(unittest.TestCase):

    def setUp(self):
        self.filename = 'test.json'

    def test_read(self):
        with patch('os.path.exists', new=MagicMock(return_value=False)):
            assert JSONFile.read(self.filename) is None

        with patch('os.path.exists', new=MagicMock(return_value=True)):
            with patch('__builtin__.open', mock_open(read_data='[]')):
                assert JSONFile.read(self.filename) == []

    def test_write(self):
        with patch('__builtin__.open', mock_open()):
            JSONFile.write([], self.filename)
