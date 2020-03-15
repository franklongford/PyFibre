from unittest import TestCase
import os
from tempfile import NamedTemporaryFile

import numpy as np

from pyfibre.tests.fixtures import test_image_path

from .. utilities import (
    parse_files, parse_file_path, pop_under_recursive,
    pop_dunder_recursive, numpy_to_python_recursive,
    python_to_numpy_recursive, replace_ext, save_json,
    load_json)


class TestUtilities(TestCase):

    def setUp(self):
        self.file_name = test_image_path
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

    def test_numpy_to_python_recursive(self):

        test_dict = {
            'int64': np.int64(0),
            'array': np.array([0, 1, 2]),
            'nested dict': {
                'id': np.int64(0),
                'list': [2, 3, 4]
            },
            'list dictionary': [
                {'array': np.array([0, 1, 2])},
                {'id': np.int(0)}
            ]
        }

        serialized_dict = numpy_to_python_recursive(test_dict)

        self.assertDictEqual(
            {
                'int64': 0,
                'array': [0, 1, 2],
                'nested dict': {
                    'id': 0,
                    'list': [2, 3, 4]
                },
                'list dictionary': [
                    {'array': [0, 1, 2]},
                    {'id': 0}
                ]
            },
            serialized_dict
        )

    def test_python_to_numpy_recursive(self):

        test_dict = {
            'int64': 0,
            'xy': [0, 1, 2],
            'nested dict': {
                'id': 0,
                'list': [2, 3, 4]
            },
            'list dictionary': [
                {'direction': [0, 1, 2]},
                {'id': 0}
            ]
        }

        deserialized_dict = python_to_numpy_recursive(test_dict)

        self.assertIsInstance(
            deserialized_dict['xy'], np.ndarray)
        self.assertIsInstance(
            deserialized_dict['list dictionary'][0]['direction'],
            np.ndarray)

    def test_replace_ext(self):

        file_name = 'some/path/to/file'

        self.assertEqual(
            'some/path/to/file.json',
            replace_ext(file_name, 'json')
        )
        self.assertEqual(
            'some/path/to/file.json',
            replace_ext(file_name + '.csv', 'json')
        )

    def test_save_load_json(self):
        data = {
            'an integer': 0,
            'a list': [0, 1, 2],
            'a dictionary': {'some key': 'some value'}
        }

        with NamedTemporaryFile() as temp_file:

            save_json(data, temp_file.name)

            self.assertTrue(os.path.exists(
                temp_file.name + '.json'))

            test_data = load_json(temp_file.name)

        self.assertDictEqual(data, test_data)

        # Test non-duplicated extensions
        with NamedTemporaryFile() as temp_file:

            save_json(data, temp_file.name + '.json')
            self.assertTrue(os.path.exists(
                temp_file.name + '.json'))
