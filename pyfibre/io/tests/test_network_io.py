import os
from unittest import TestCase

import networkx as nx

from pyfibre.io.network_io import save_network, load_network


class TestNetworkIO(TestCase):

    def setUp(self):

        self.network = nx.Graph()
        self.network.add_nodes_from([1, 2])
        self.network.add_edge(1, 2)

    def test_save_network(self):

        try:
            save_network(self.network, 'test', 'graph')
            self.assertTrue(os.path.exists('test_graph.pkl'))
        finally:
            if os.path.exists('test_graph.pkl'):
                os.remove('test_graph.pkl')

    def test_load_network(self):

        try:
            save_network(self.network, 'test', 'graph')
            network = load_network('test', 'graph')

            self.assertEqual(
                self.network.nodes,
                network.nodes)

            self.assertEqual(
                self.network.edges,
                network.edges)
        finally:
            if os.path.exists('test_graph.pkl'):
                os.remove('test_graph.pkl')