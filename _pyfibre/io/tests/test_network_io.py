import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

from pyfibre.io.network_io import save_network, load_network
from pyfibre.tests.probe_classes.utilities import generate_probe_graph


class TestNetworkIO(TestCase):

    def setUp(self):

        self.network = generate_probe_graph()

    def test_save_network(self):

        with NamedTemporaryFile() as tmp_file:
            save_network(
                self.network, tmp_file.name, 'graph')
            self.assertTrue(
                os.path.exists(f'{tmp_file.name}_graph.pkl'))

    def test_load_network(self):

        with NamedTemporaryFile() as tmp_file:
            save_network(
                self.network, tmp_file.name, 'graph')

            network = load_network(tmp_file.name, 'graph')

        self.assertListEqual(
            list(self.network.nodes), list(network.nodes))

        self.assertListEqual(
            list(self.network.edges), list(network.edges))
