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

        new_fibre = Fibre(**status)
        status = new_fibre.__getstate__()

        self.assertDictEqual(
            status['graph'],
            {'directed': False,
             'graph': {},
             'links': [{'r': 1.4142135623730951, 'source': 2, 'target': 3},
                       {'r': 1.4142135623730951, 'source': 3, 'target': 4},
                       {'r': 1, 'source': 4, 'target': 5}],
             'multigraph': False,
             'nodes': [{'xy': [0, 0], 'id': 2},
                       {'xy': [1, 1], 'id': 3},
                       {'xy': [2, 2], 'id': 4},
                       {'xy': [2, 3], 'id': 5}]
             }
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
        self.assertEqual(11, len(database))

        fibre.image = image

        database = fibre.generate_database()
        self.assertEqual(11, len(database))
