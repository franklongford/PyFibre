from tempfile import NamedTemporaryFile
from unittest import TestCase
import os

from pyfibre.io.object_io import (
    save_pyfibre_object, load_pyfibre_object,
    save_pyfibre_objects, load_pyfibre_objects,
    save_fibres, load_fibres,
    save_fibre_networks, load_fibre_networks,
    save_fibre_segments, load_fibre_segments,
    save_cell_segments, load_cell_segments
)
from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.objects.segments import (
    FibreSegment, CellSegment)
from pyfibre.tests.probe_classes.objects import (
    ProbeGraphSegment, ProbeSegment, ProbeFibre,
    ProbeFibreNetwork)
from pyfibre.utilities import NotSupportedError


class TestPyFibreObjectIO(TestCase):

    def setUp(self):

        self.graph_segment = ProbeGraphSegment()
        self.segment = ProbeSegment()

    def test_exception_handling(self):

        with NamedTemporaryFile() as temp_file:

            with self.assertRaises(AttributeError):
                save_pyfibre_object(
                    self.segment, temp_file.name,
                    mode='not_supported')

            with self.assertRaises(NotSupportedError):
                save_pyfibre_object(
                    self.segment, temp_file.name,
                    mode='json')

    def test_save_load_pyfibre_object(self):

        with NamedTemporaryFile() as temp_file:
            save_pyfibre_object(
                self.segment, temp_file.name,
                mode='array')
            self.assertTrue(
                os.path.exists(f'{temp_file.name}.npy'))

            test_segment = load_pyfibre_object(
                f'{temp_file.name}.npy', ProbeSegment,
                mode='array'
            )

            self.assertIsInstance(test_segment, ProbeSegment)
            self.assertEqual(
                self.segment.region.bbox,
                test_segment.region.bbox
            )

        with NamedTemporaryFile() as temp_file:
            save_pyfibre_object(
                self.graph_segment, temp_file.name,
                mode='json')
            self.assertTrue(
                os.path.exists(f'{temp_file.name}.json'))

            test_segment = load_pyfibre_object(
                f'{temp_file.name}.json', ProbeGraphSegment,
                mode='json'
            )

            self.assertIsInstance(test_segment, ProbeGraphSegment)
            self.assertEqual(
                self.graph_segment.graph.number_of_nodes(),
                test_segment.graph.number_of_nodes()
            )
            self.assertEqual(
                self.graph_segment.graph.number_of_edges(),
                test_segment.graph.number_of_edges()
            )

    def test_save_load_pyfibre_objects(self):

        with NamedTemporaryFile() as temp_file:
            save_pyfibre_objects(
                [self.segment, self.segment], temp_file.name,
                mode='array', shape=(10, 10))
            self.assertTrue(
                os.path.exists(f'{temp_file.name}.npy'))

            test_segments = load_pyfibre_objects(
                f'{temp_file.name}.npy', ProbeSegment,
                mode='array'
            )

            self.assertEqual(2, len(test_segments))
            self.assertIsInstance(
                test_segments[0], ProbeSegment)
            self.assertEqual(
                self.segment.region.bbox,
                test_segments[0].region.bbox
            )

        with NamedTemporaryFile() as temp_file:
            save_pyfibre_objects(
                [self.graph_segment, self.graph_segment],
                temp_file.name, mode='json')
            self.assertTrue(
                os.path.exists(f'{temp_file.name}.json'))

            test_segments = load_pyfibre_objects(
                f'{temp_file.name}.json', ProbeGraphSegment,
                mode='json'
            )

            self.assertEqual(2, len(test_segments))
            self.assertIsInstance(
                test_segments[0], ProbeGraphSegment)
            self.assertEqual(
                self.graph_segment.graph.number_of_nodes(),
                test_segments[0].graph.number_of_nodes()
            )
            self.assertEqual(
                self.graph_segment.graph.number_of_edges(),
                test_segments[0].graph.number_of_edges()
            )


class TestObjectIO(TestCase):

    def setUp(self):
        self.segment = ProbeSegment()
        self.fibre = ProbeFibre()
        self.fibre_network = ProbeFibreNetwork()

    def test_save_load_fibres(self):

        with NamedTemporaryFile() as temp_file:
            save_fibres([self.fibre, self.fibre], temp_file.name)
            self.assertTrue(
                os.path.exists(f'{temp_file.name}_fibres.json'))

            test_fibres = load_fibres(
                f'{temp_file.name}_fibres.json'
            )
            self.assertEqual(2, len(test_fibres))
            self.assertIsInstance(
                test_fibres[0], Fibre)
            self.assertEqual(
                self.fibre.graph.number_of_nodes(),
                test_fibres[0].graph.number_of_nodes()
            )
            self.assertEqual(
                self.fibre.graph.number_of_edges(),
                test_fibres[0].graph.number_of_edges()
            )

    def test_save_load_fibre_networks(self):

        with NamedTemporaryFile() as temp_file:
            save_fibre_networks(
                [self.fibre_network, self.fibre_network],
                temp_file.name)
            self.assertTrue(
                os.path.exists(f'{temp_file.name}_fibre_networks.json'))

            test_fibre_networks = load_fibre_networks(
                f'{temp_file.name}_fibre_networks.json'
            )
            self.assertEqual(2, len(test_fibre_networks))
            self.assertIsInstance(
                test_fibre_networks[0], FibreNetwork)
            self.assertEqual(
                self.fibre.graph.number_of_nodes(),
                test_fibre_networks[0].graph.number_of_nodes()
            )
            self.assertEqual(
                self.fibre.graph.number_of_edges(),
                test_fibre_networks[0].graph.number_of_edges()
            )

    def test_save_load_fibre_segments(self):

        with NamedTemporaryFile() as temp_file:
            save_fibre_segments(
                [self.segment, self.segment],
                temp_file.name, shape=(10, 10))
            self.assertTrue(
                os.path.exists(f'{temp_file.name}_fibre_segments.npy'))

            test_fibre_segments = load_fibre_segments(
                f'{temp_file.name}_fibre_segments.npy'
            )
            self.assertEqual(2, len(test_fibre_segments))
            self.assertIsInstance(
                test_fibre_segments[0], FibreSegment)
            self.assertEqual(
                self.segment.region.bbox,
                test_fibre_segments[0].region.bbox
            )

    def test_save_load_cell_segments(self):

        with NamedTemporaryFile() as temp_file:
            save_cell_segments(
                [self.segment, self.segment],
                temp_file.name, shape=(10, 10))
            self.assertTrue(
                os.path.exists(f'{temp_file.name}_cell_segments.npy'))

            test_cell_segments = load_cell_segments(
                f'{temp_file.name}_cell_segments.npy'
            )
            self.assertEqual(2, len(test_cell_segments))
            self.assertIsInstance(
                test_cell_segments[0], CellSegment)
            self.assertEqual(
                self.segment.region.bbox,
                test_cell_segments[0].region.bbox
            )
