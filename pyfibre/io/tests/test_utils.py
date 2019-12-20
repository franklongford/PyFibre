from unittest import mock, TestCase
import os

from .. utils import (
    parse_files, parse_file_path, pop_under_recursive,
    pop_dunder_recursive)

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestUtils(TestCase):

    def setUp(self):
        self.file_name = (
            pyfibre_dir + '/tests/fixtures/test-pyfibre-pl-shg-Stack.tif'
        )
        self.directory = os.path.dirname(self.file_name)
        self.key = 'shg'
        self.test_dict = {
            '__traits_version__': '4.6.0',
            'some_important_data':
                {'__traits_version__': '4.6.0', 'value': 10},
            '_some_private_data':
                {'__instance_traits__': ['yes', 'some']},
            '___':
                {'__': 'a', 'foo': 'bar'},
            'list_of_dicts': [
                {'__bad_key__': 'bad', 'good_key': 'good'},
                {'also_good_key': 'good'}]
        }

    def test_parse_file_path(self):

        file_name, directory = parse_file_path(
            '/a/path/to/some/file-pl-shg.tif')

        self.assertIsNone(file_name)
        self.assertIsNone(directory)

        file_name, directory = parse_file_path(self.file_name)
        self.assertEqual(self.file_name, file_name)
        self.assertIsNone(directory)

        file_name, directory = parse_file_path(self.directory)
        self.assertIsNone(file_name)
        self.assertEqual(self.directory, directory)

    def test_parse_files(self):

        input_files = parse_files(self.file_name, key=self.key)
        self.assertEqual(input_files[0], self.file_name)

        input_files = parse_files(directory=self.directory,
                                  key=self.key)
        self.assertEqual(input_files[0], self.file_name)

    def test_under_recursive(self):

        expected = {
            'some_important_data': {'value': 10},
            'list_of_dicts': [
                {'good_key': 'good'},
                {'also_good_key': 'good'}]
        }
        self.assertEqual(pop_under_recursive(self.test_dict), expected)

    def test_dunder_recursive(self):

        expected = {'some_important_data': {'value': 10},
                    '_some_private_data': {},
                    'list_of_dicts': [{'good_key': 'good'},
                                      {'also_good_key': 'good'}]
                    }
        self.assertEqual(pop_dunder_recursive(self.test_dict), expected)
