from unittest import TestCase

import numpy as np

from pyfibre.model.objects.fibre import (
    Fibre
)
from pyfibre.tests.probe_classes import generate_probe_graph


class TestFibre(TestCase):

    def setUp(self):

        self.graph = generate_probe_graph()

    def test__getstate__(self):
        fibre = Fibre(graph=self.graph)

        status = fibre.__getstate__()

        self.assertListEqual(
            ['graph', 'growing'],
            list(status.keys())
        )

    def test_node_list_init(self):

        fibre = Fibre(nodes=[2, 3, 4, 5],
                      edges=[(3, 2), (3, 4), (4, 5)])

        self.assertEqual(4, fibre.number_of_nodes)
        self.assertEqual([2, 3, 4, 5], fibre.node_list)
        self.assertTrue(fibre.growing)

        self.assertTrue(np.allclose(np.array([0, 0]), fibre._d_coord))
        self.assertTrue(np.allclose(np.array([0, 0]), fibre.direction))
        self.assertEqual(90, fibre.angle)
        self.assertEqual(0, fibre.euclid_l)
        self.assertEqual(0, fibre.fibre_l)
        self.assertTrue(np.isnan(fibre.waviness))

    def test_network_init(self):

        fibre = Fibre(graph=self.graph)

        self.assertTrue(fibre.growing)
        self.assertTrue(np.allclose(np.array([2, 3]), fibre._d_coord))
        self.assertTrue(np.allclose(
            np.array([-0.5547002, -0.83205029]), fibre.direction))
        self.assertAlmostEqual(146.30993247, fibre.angle)
        self.assertAlmostEqual(3.60555127, fibre.euclid_l)
        self.assertAlmostEqual(3.82842712, fibre.fibre_l)
        self.assertAlmostEqual(0.94178396, fibre.waviness)

    def test_generate_database(self):
        fibre = Fibre(graph=self.graph)

        database = fibre.generate_database()
        self.assertEqual(11, len(database))

        image = np.ones((5, 5))
        image[2:, 2:] = 2

        database = fibre.generate_database(image)
        self.assertEqual(25, len(database))

        fibre.image = image

        database = fibre.generate_database()
        self.assertEqual(25, len(database))
