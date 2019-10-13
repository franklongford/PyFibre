from unittest import TestCase

import numpy as np

from pyfibre.model.tools.fibre_assignment import (
    Fibre
)
from pyfibre.tests.probe_classes import generate_probe_network


class TestFibre(TestCase):

    def setUp(self):

        self.graph = generate_probe_network()

    def test_node_list_init(self):

        fibre = Fibre(nodes=[2, 3, 4, 5],
                      edges=[(3, 2), (3, 4), (4, 5)])

        self.assertEqual([2, 3, 4, 5], list(fibre.nodes))
        self.assertTrue(fibre.growing)
        self.assertTrue(np.allclose(np.array([0, 0]), fibre._d_coord))
        self.assertTrue(np.allclose(np.array([0, 0]), fibre.direction))
        self.assertEqual(0, fibre.euclid_l)
        self.assertEqual(0, fibre.fibre_l)
        self.assertTrue(np.isnan(fibre.waviness))

    def test_network_init(self):

        fibre = Fibre(graph=self.graph)

        self.assertEqual([2, 3, 4, 5], list(fibre.nodes))
        self.assertTrue(
            np.allclose(np.array([1, 1]), fibre.nodes[3]['xy'])
        )
        self.assertAlmostEqual(np.sqrt(2), fibre.edges[3, 4]['r'])
        self.assertTrue(fibre.growing)
        self.assertTrue(np.allclose(np.array([2, 3]), fibre._d_coord))
        self.assertTrue(np.allclose(
            np.array([-0.5547002, -0.83205029]), fibre.direction))

        self.assertAlmostEqual(3.60555127, fibre.euclid_l)
        self.assertAlmostEqual(3.82842712, fibre.fibre_l)
        self.assertAlmostEqual(0.94178396, fibre.waviness)

