from unittest import TestCase

import networkx as nx
from skimage.io import imread

from pyfibre.model.tools.network_extraction import (
    build_network, clean_network, fibre_network_assignment
)
from pyfibre.tests.fixtures import test_shg_image_path
from pyfibre.tests.probe_classes.utilities import (
    generate_probe_graph
)


class TestExtraction(TestCase):

    def setUp(self):
        self.network = generate_probe_graph()
        self.image = imread(test_shg_image_path).mean(axis=-1)

    def test_build_network(self):

        network = build_network(self.image)
        self.assertFalse(list(nx.isolates(network)))
        self.assertEqual(575, network.number_of_nodes())
        self.assertEqual(621, network.number_of_edges())

    def test_clean_network(self):

        network = clean_network(self.network, r_thresh=1)
        self.assertListEqual([0, 1, 2, 3], list(network.nodes))

        network = clean_network(self.network, r_thresh=2)
        self.assertListEqual([], list(network.nodes))

    def test_fibre_network_assignment(self):

        fibre_networks = fibre_network_assignment(
            self.network)
        fibres = fibre_networks[0].fibres

        self.assertEqual(1, len(fibre_networks))
        self.assertEqual(1, len(fibres))

        self.assertListEqual(
            [2, 3, 4, 5], fibre_networks[0].node_list
        )
        self.assertListEqual(
            [0, 1], list(fibre_networks[0].red_graph.nodes)
        )
        self.assertListEqual(
            [0, 1, 2, 3], fibres[0].node_list
        )
