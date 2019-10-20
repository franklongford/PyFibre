from unittest import TestCase

import networkx as nx

from pyfibre.io.tif_reader import load_image
from pyfibre.model.tools.extraction import (
    build_network, clean_network, simplify_network,
    network_extraction
)
from pyfibre.tests.test_utilities import test_image_path
from pyfibre.tests.probe_classes import (
    generate_probe_network
)


class TestExtraction(TestCase):

    def setUp(self):
        self.network = generate_probe_network()
        self.image = load_image(test_image_path)[0].mean(axis=-1)

    def test_build_network(self):

        network = build_network(self.image)
        self.assertFalse(list(nx.isolates(network)))
        self.assertEqual(526, network.number_of_nodes())
        self.assertEqual(577, network.number_of_edges())

    def test_clean_network(self):

        network = clean_network(self.network, r_thresh=1)
        self.assertListEqual([0, 1, 2, 3], list(network.nodes))

        network = clean_network(self.network, r_thresh=2)
        self.assertListEqual([], list(network.nodes))

    def test_simplify_network(self):

        network = simplify_network(self.network)
        self.assertListEqual([0, 1], list(network.nodes))

    def test_network_extraction(self):

        networks, networks_red, fibres = network_extraction(self.network)

        self.assertEqual(1, len(networks))
        self.assertEqual(1, len(networks_red))
        self.assertEqual(1, len(fibres))

        self.assertListEqual(
            [2, 3, 4, 5], list(networks[0].nodes)
        )
        self.assertListEqual(
            [0, 1], list(networks_red[0].nodes)
        )
        self.assertListEqual(
            [0, 1, 2, 3], list(fibres[0][0].nodes)
        )
