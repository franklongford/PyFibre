from unittest import TestCase

import numpy as np

from pyfibre.model.objects.fibre_network import (
    FibreNetwork
)
from pyfibre.tests.probe_classes import ProbeFibreNetwork


class TestFibreNetwork(TestCase):

    def setUp(self):

        self.network = ProbeFibreNetwork()

    def test__getstate__(self):

        state = self.network.__getstate__()

        self.assertIn('fibres', state)
        self.assertIn('red_graph', state)

        self.assertListEqual(
            [], state['fibres']
        )

        self.assertDictEqual(
            state['red_graph'],
            {'directed': False,
             'multigraph': False,
             'graph': {},
             'nodes': [{'xy': [0, 0], 'id': 0},
                       {'xy': [2, 3], 'id': 1}],
             'links': [{'r': 3.605551275463989, 'source': 0, 'target': 1}]}
        )

    def test_deserialise(self):
        status = self.network.__getstate__()
        new_network = FibreNetwork(**status)
        status = new_network.__getstate__()

        self.assertDictEqual(
            status['red_graph'],
            {'directed': False,
             'multigraph': False,
             'graph': {},
             'nodes': [{'xy': [0, 0], 'id': 0},
                       {'xy': [2, 3], 'id': 1}],
             'links': [{'r': 3.605551275463989, 'source': 0, 'target': 1}]}
        )

    def test_fibres(self):

        fibres = self.network.generate_fibres()

        self.assertEqual(1, len(fibres))
        self.assertListEqual([0, 1, 2, 3], fibres[0].node_list)

        self.network.fibres = fibres
        status = self.network.__getstate__()
        self.assertEqual(1, len(status["fibres"]))

    def test_serialisation(self):
        self.network.fibres = self.network.generate_fibres()
        status = self.network.__getstate__()
        new_network = FibreNetwork(**status)

        self.assertEqual(1, len(new_network.fibres))
        self.assertListEqual(
            [0, 1, 2, 3], new_network.fibres[0].node_list)

    def test_generate_database(self):

        database = self.network.generate_database()
        self.assertEqual(12, len(database))

        image = np.ones((10, 10))
        image[2:, 2:] = 2

        database = self.network.generate_database(image)
        self.assertEqual(26, len(database))

        self.network.image = image

        database = self.network.generate_database()
        self.assertEqual(26, len(database))