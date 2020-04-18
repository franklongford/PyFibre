from unittest import TestCase
import os
from tempfile import NamedTemporaryFile

import networkx as nx
import numpy as np

from pyfibre.tests.fixtures import test_shg_pl_trans_image_path
from pyfibre.tests.probe_classes import generate_probe_graph

from .. utilities import (
    parse_file_path, pop_under_recursive,
    pop_dunder_recursive, numpy_to_python_recursive,
    python_to_numpy_recursive, replace_ext, save_json,
    load_json, serialize_networkx_graph, deserialize_networkx_graph,
    check_file_name, check_string
)


class TestUtilities(TestCase):

    def setUp(self):
        self.file_name = test_shg_pl_trans_image_path
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
        self.graph = generate_probe_graph()

    def test_parse_files(self):

        input_files = parse_file_path(
            self.file_name, key=self.key)
        self.assertEqual(1, len(input_files))
        self.assertIn(self.file_name, input_files)

        input_files = parse_file_path(
            self.directory, key=self.key)
        self.assertEqual(2, len(input_files))
        self.assertIn(self.file_name, input_files)

        input_files = parse_file_path(
            self.directory, key='not-there')
        self.assertListEqual([], input_files)

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

    def test_serialize_networkx_graph(self):

        data = serialize_networkx_graph(self.graph)

        self.assertDictEqual(
            data,
            {'directed': False,
             'graph': {},
             'links': [{'r': 1.4142135623730951, 'source': 2, 'target': 3},
                       {'r': 1.4142135623730951, 'source': 3, 'target': 4},
                       {'r': 1, 'source': 4, 'target': 5}],
             'multigraph': False,
             'nodes': [{'xy': [0, 0], 'id': 2},
                       {'xy': [1, 1], 'id': 3},
                       {'xy': [2, 2], 'id': 4},
                       {'xy': [2, 3], 'id': 5}]
             }
        )

    def test_deserialize_networkx_graph(self):

        data = {
            'directed': False,
            'graph': {},
            'links': [{'r': 1.4142135623730951, 'source': 2, 'target': 3},
                      {'r': 1.4142135623730951, 'source': 3, 'target': 4},
                      {'r': 1, 'source': 4, 'target': 5}],
            'multigraph': False,
            'nodes': [{'xy': [0, 0], 'id': 2},
                      {'xy': [1, 1], 'id': 3},
                      {'xy': [2, 2], 'id': 4},
                      {'xy': [2, 3], 'id': 5}]
        }

        graph = deserialize_networkx_graph(data)

        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(4, graph.number_of_nodes())
        self.assertIsInstance(graph.nodes[2]['xy'], np.ndarray)

    def test_string_functions(self):
        string = "/dir/folder/test_file_SHG.pkl"

        self.assertEqual(
            "/dir/test_file_SHG.pkl",
            check_string(string, -2, '/', 'folder'))
        self.assertEqual(
            "/dir/folder/test_file",
            check_file_name(string, 'SHG', 'pkl'))
