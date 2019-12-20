from unittest import TestCase
import os

from pyfibre.io.object_io import (
    save_base_graph_segment, load_base_graph_segment)
from pyfibre.model.objects.base_graph_segment import (
    BaseGraphSegment
)
from pyfibre.tests.probe_classes import generate_probe_graph


class TestObjectIO(TestCase):

    def setUp(self):

        self.graph = generate_probe_graph()
        self.graph_segment = BaseGraphSegment(graph=self.graph)

    def test_save_load_base_graph_segment(self):

        try:
            save_base_graph_segment(self.graph_segment, 'test_json')
            self.assertTrue(os.path.exists('test_json.json'))

            test_segment = load_base_graph_segment('test_json')

            self.assertIsInstance(test_segment, BaseGraphSegment)
            self.assertEqual(
                self.graph.number_of_nodes(),
                test_segment.graph.number_of_nodes()
            )

            self.assertEqual(
                self.graph.number_of_edges(),
                test_segment.graph.number_of_edges()
            )

        finally:
            if os.path.exists('test_json.json'):
                os.remove('test_json.json')
