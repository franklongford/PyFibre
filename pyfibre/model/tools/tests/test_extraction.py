from unittest import TestCase
import numpy as np
import networkx as nx

from pyfibre.io.tif_reader import load_image
from pyfibre.model.tools.extraction import (
    build_network, simplify_network, network_extraction
)
from pyfibre.tests.test_utilities import test_image_path


class TestExtraction(TestCase):

    def setUp(self):
        self.image = load_image(test_image_path)[0].mean(axis=-1)

    def test_build_network(self):

        network = build_network(self.image)
        self.assertFalse(list(nx.isolates(network)))
        self.assertEqual(526, network.number_of_nodes())
        self.assertEqual(579, network.number_of_edges())

    def test_clean_network(self):
        pass

    def test_simplify_network(self):
        pass

    def test_network_extraction(self):
        pass