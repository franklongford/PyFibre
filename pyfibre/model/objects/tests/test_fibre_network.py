from unittest import TestCase

from pyfibre.model.objects.fibre_network import (
    FibreNetwork
)
from pyfibre.model.tools.metrics import (
    FIBRE_METRICS, NETWORK_METRICS)
from pyfibre.tests.probe_classes import ProbeFibreNetwork, ProbeFibre


class TestFibreNetwork(TestCase):

    def setUp(self):

        self.network = ProbeFibreNetwork()
        self.fibres = [ProbeFibre(), ProbeFibre(), ProbeFibre()]

    def test_to_json(self):

        state = self.network.to_json()

        self.assertIn('fibres', state)
        self.assertIn('red_graph', state)
        self.assertListEqual([], state['fibres'])
        self.assertIsNone(state['red_graph'])

        self.network.fibres = self.fibres
        state = self.network.to_json()
        self.assertEqual(3, len(state['fibres']))

        self.network.red_graph = self.network.generate_red_graph()
        state = self.network.to_json()

        self.assertDictEqual(
            state['red_graph'],
            {
                'directed': False,
                'multigraph': False,
                'graph': {},
                'nodes': [{'xy': [0, 0], 'id': 0},
                          {'xy': [2, 3], 'id': 1}],
                'links': [{'r': 3.605551275463989,
                           'source': 0,
                           'target': 1}]
             }
        )

    def test_from_json(self):

        self.network.red_graph = self.network.generate_red_graph()
        self.network.fibres = self.fibres

        status = self.network.to_json()
        new_network = FibreNetwork.from_json(status)
        status = new_network.to_json()

        self.assertDictEqual(
            status['red_graph'],
            {
                'directed': False,
                'multigraph': False,
                'graph': {},
                'nodes': [{'xy': [0, 0], 'id': 0},
                          {'xy': [2, 3], 'id': 1}],
                'links': [{'r': 3.605551275463989,
                           'source': 0,
                           'target': 1}]
            }
        )

        self.assertEqual(3, len(new_network.fibres))
        self.assertListEqual(
            [2, 3, 4, 5], new_network.fibres[0].node_list)

    def test_fibres(self):

        fibres = self.network.generate_fibres()

        self.assertEqual(1, len(fibres))
        self.assertListEqual([0, 1, 2, 3], fibres[0].node_list)

        self.network.fibres = fibres
        status = self.network.to_json()
        self.assertEqual(1, len(status["fibres"]))

    def test_generate_database(self):

        self.network.red_graph = self.network.generate_red_graph()
        self.network.fibres = self.fibres

        database = self.network.generate_database()
        self.assertEqual(8, len(database))

        self.assertIn('Fibre Angle SDI', database)

        for metric in FIBRE_METRICS:
            self.assertIn(f'Mean Fibre {metric}', database)

        for metric in NETWORK_METRICS:
            self.assertIn(f'Fibre Network {metric}', database)
