from unittest import TestCase

import numpy as np

from pyfibre.fibers.fibre import Fibre, fiber_from_node_list
from pyfibre.testing.example_objects import generate_probe_graph


class TestFibre(TestCase):
    def setUp(self):
        self.fibre = Fibre(graph=generate_probe_graph())

    def test_network_init(self):
        self.assertTrue(self.fibre.growing)
        self.assertTrue(np.allclose(np.array([2, 3]), self.fibre._d_coord))
        self.assertTrue(
            np.allclose(np.array([-0.5547002, -0.83205029]), self.fibre.direction)
        )
        self.assertAlmostEqual(146.30993247, self.fibre.angle)
        self.assertAlmostEqual(3.60555127, self.fibre.euclid_l)
        self.assertAlmostEqual(3.82842712, self.fibre.fibre_l)
        self.assertAlmostEqual(0.94178396, self.fibre.waviness)


class TestFibreFromNodeList(TestCase):
    def test_node_list_init(self):
        fibre = fiber_from_node_list(nodes=[2, 3, 4, 5], edges=[(3, 2), (3, 4), (4, 5)])

        self.assertEqual(4, fibre.graph.number_of_nodes())
        self.assertEqual([2, 3, 4, 5], fibre.node_list)
        self.assertTrue(fibre.growing)

        self.assertTrue(np.allclose(np.array([0, 0]), fibre._d_coord))
        self.assertTrue(np.allclose(np.array([0, 0]), fibre.direction))
        self.assertEqual(90, fibre.angle)
        self.assertEqual(0, fibre.euclid_l)
        self.assertEqual(0, fibre.fibre_l)
        self.assertTrue(np.isnan(fibre.waviness))
