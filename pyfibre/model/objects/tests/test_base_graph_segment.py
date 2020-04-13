from unittest import TestCase

import numpy as np

from pyfibre.tests.probe_classes import generate_probe_graph
from pyfibre.tests.dummy_classes import DummyGraphSegment
from pyfibre.tests.probe_classes import ProbeGraphSegment


class TestBaseGraphSegment(TestCase):

    def setUp(self):

        self.graph = generate_probe_graph()
        self.graph_segment = ProbeGraphSegment()

    def test__getstate__(self):

        status = self.graph_segment.to_json()

        self.assertIn('shape', status)
        self.assertDictEqual(
            status['graph'],
            {'directed': False,
             'graph': {},
             'links': [{'r': 1.4142135623730951,
                        'source': 2, 'target': 3},
                       {'r': 1.4142135623730951,
                        'source': 3, 'target': 4},
                       {'r': 1, 'source': 4, 'target': 5}],
             'multigraph': False,
             'nodes': [{'xy': [0, 0], 'id': 2},
                       {'xy': [1, 1], 'id': 3},
                       {'xy': [2, 2], 'id': 4},
                       {'xy': [2, 3], 'id': 5}]
             }
        )

    def test_deserialise(self):
        status = self.graph_segment.to_json()
        new_graph_segment = DummyGraphSegment.from_json(status)
        status = new_graph_segment.to_json()

        self.assertDictEqual(
            status['graph'],
            {'directed': False,
             'graph': {},
             'links': [{'r': 1.4142135623730951,
                        'source': 2, 'target': 3},
                       {'r': 1.4142135623730951,
                        'source': 3, 'target': 4},
                       {'r': 1, 'source': 4, 'target': 5}],
             'multigraph': False,
             'nodes': [{'xy': [0, 0], 'id': 2},
                       {'xy': [1, 1], 'id': 3},
                       {'xy': [2, 2], 'id': 4},
                       {'xy': [2, 3], 'id': 5}]
             }
        )

    def test_network_init(self):

        self.assertEqual(4, self.graph_segment.number_of_nodes)
        self.assertListEqual(
            [2, 3, 4, 5], self.graph_segment.node_list)
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

        segment = self.graph_segment.region
        self.assertEqual(
            (3, 4), self.graph_segment.region.image.shape)
        self.assertEqual(12, segment.area)

        with self.assertRaises(AttributeError):
            _ = segment.intensity_image

        self.graph_segment._iterations = 0
        self.graph_segment._area_threshold = 0
        self.graph_segment._sigma = None
        segment = self.graph_segment.region

        self.assertEqual((3, 4), segment.image.shape)
        self.assertEqual(4, segment.area)

        self.graph_segment.image = np.ones((5, 5)) * 2
        segment = self.graph_segment.region

        self.assertEqual((3, 4), segment.image.shape)
        self.assertEqual((3, 4), segment.intensity_image.shape)

    def test_add_node_edge(self):

        self.graph_segment.add_node(6)

        self.assertEqual(5, self.graph_segment.number_of_nodes)

        self.graph_segment.add_edge(6, 2)

        self.assertEqual(4, self.graph_segment.graph.size())
