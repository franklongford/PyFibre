from unittest import TestCase

from pyfibre.fibers.fibre_network import FibreNetwork

from pyfibre.testing.example_objects import generate_probe_graph


class TestFibreNetwork(TestCase):
    def setUp(self):
        self.network = FibreNetwork(graph=generate_probe_graph())

    def test_fibres(self):
        fibres = self.network.generate_fibres()

        self.assertEqual(1, len(fibres))
        self.assertListEqual([0, 1, 2, 3], fibres[0].node_list)
