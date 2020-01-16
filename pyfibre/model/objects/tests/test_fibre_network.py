from unittest import TestCase

import numpy as np

from pyfibre.model.objects.fibre_network import (
    FibreNetwork
)
from pyfibre.tests.probe_classes import generate_probe_graph


class TestNetwork(TestCase):

    def setUp(self):

        self.graph = generate_probe_graph()
        self.network = FibreNetwork(graph=self.graph)

    def test__getstate__(self):

        status = self.network.__getstate__()

        self.assertListEqual(
            ['graph'],
            list(status.keys())
        )

    def test_fibres(self):

        fibres = self.network.fibres

        self.assertEqual(1, len(fibres))
        self.assertListEqual([0, 1, 2, 3], fibres[0].node_list)

    def test_generate_database(self):

        database = self.network.generate_database()
        self.assertEqual(12, len(database))

        image = np.ones((5, 5))
        image[2:, 2:] = 2

        database = self.network.generate_database(image)
        self.assertEqual(26, len(database))

        self.network.image = image

        database = self.network.generate_database()
        self.assertEqual(26, len(database))