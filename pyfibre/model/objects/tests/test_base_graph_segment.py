from unittest import TestCase

import numpy as np

from pyfibre.model.objects.base_graph_segment import (
    BaseGraphSegment
)
from pyfibre.tests.probe_classes import generate_probe_graph


class TestBaseGraphSegment(TestCase):

    def setUp(self):

        self.graph = generate_probe_graph()
        self.graph_segment = BaseGraphSegment(graph=self.graph)

    def test_network_init(self):

        self.assertEqual(4, self.graph_segment.number_of_nodes)
        self.assertListEqual([2, 3, 4, 5], self.graph_segment.node_list)
        self.assertEqual(3, self.graph_segment.graph.size())

        self.assertTrue(
            np.allclose(np.array([1, 1]),
                        self.graph_segment.graph.nodes[3]['xy']))
        self.assertAlmostEqual(
            np.sqrt(2), self.graph_segment.graph.edges[3, 4]['r'])

        self.assertTrue(np.allclose(
            np.array([[0, 0],
                      [1, 1],
                      [2, 2],
                      [2, 3]]),
            self.graph_segment.node_coord))

    def test_network_segment(self):

        self.assertEqual((3, 4), self.graph_segment.segment.image.shape)
        self.assertEqual(12, self.graph_segment.segment.area)

        with self.assertRaises(AttributeError):
            _ = self.graph_segment.segment.intensity_image

        self.graph_segment._iterations = 0
        self.graph_segment._area_threshold = 0

        self.assertEqual((3, 4), self.graph_segment.segment.image.shape)
        self.assertEqual(4, self.graph_segment.segment.area)

        self.graph_segment.image = np.ones((5, 5)) * 2

        self.assertEqual(
            (3, 4), self.graph_segment.segment.image.shape)
        self.assertEqual(
            (3, 4), self.graph_segment.segment.intensity_image.shape)

    def test_add_node_edge(self):

        self.graph_segment.add_node(6)

        self.assertEqual(5, self.graph_segment.number_of_nodes)

        self.graph_segment.add_edge(6, 2)

        self.assertEqual(4, self.graph_segment.graph.size())
