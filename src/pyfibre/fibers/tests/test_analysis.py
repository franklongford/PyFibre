import importlib.resources
from unittest import TestCase

import networkx as nx
from skimage.io import imread

from pyfibre.fibers.analysis import network_analysis


class TestNetworkAnalysis(TestCase):
    def setUp(self):
        testing_dir = self.enterContext(importlib.resources.path("pyfibre.testing"))
        self.test_shg_image_path = str(
            testing_dir / "fixtures" / "test-pyfibre-shg-Stack.tif"
        )

    def test_network_analysis(self):
        # Given
        shg_image = imread(self.test_shg_image_path).mean(axis=-1)
        shg_image = shg_image / shg_image.max()

        # When
        network = network_analysis(
            shg_image[50:, 50:],
            scale=1.25,
        )

        # Then
        self.assertTrue(network.number_of_nodes() > 0)
        self.assertTrue(network.number_of_edges() > 0)
        self.assertTrue(len(list(nx.connected_components(network))) > 0)
