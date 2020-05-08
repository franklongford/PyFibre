import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

from pyfibre.tests.probe_classes.multi_images import ProbeSHGPLTransImage
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, ProbeFibreSegment, ProbeCellSegment)

from .. image_analyser import ImageAnalyser


LOAD_NETWORK_PATH = "networkx.read_gpickle"
SAVE_NETWORK_PATH = "networkx.write_gpickle"

LOAD_JSON_PATH = "json.load"
SAVE_JSON_PATH = "json.dump"

LOAD_REGION_PATH = "numpy.load"
SAVE_REGION_PATH = "numpy.save"


def mock_load(*args, klass=None, **kwargs):
    print('mock_load called')
    return klass()


class TestImageAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.image_analyser = ImageAnalyser()
        self.fibre_networks = [ProbeFibreNetwork()]

        self.fibre_segments = [ProbeFibreSegment()]
        self.cell_segments = [ProbeCellSegment()]

    def test_get_ow_options(self):

        filename = "test_filename"

        ow_network, ow_segment, ow_metric = (
            self.image_analyser.get_analysis_options(
                self.multi_image, filename))

        self.assertTrue(ow_network)
        self.assertTrue(ow_segment)
        self.assertTrue(ow_metric)

    def test_network_analysis(self):

        with NamedTemporaryFile() as tmp_file:

            network, fibre_networks = (
                self.image_analyser.network_analysis(
                    self.multi_image, tmp_file.name))

            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre_networks.json'))

        self.assertEqual(556, network.number_of_nodes())
        self.assertEqual(578, network.number_of_edges())
        self.assertEqual(5, len(fibre_networks))

    def test_segment_analysis(self):

        with NamedTemporaryFile() as tmp_file:

            fibre_segments, cell_segments = (
                self.image_analyser.segment_analysis(
                    self.multi_image, tmp_file.name,
                    self.fibre_networks))

            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre_segments.npy'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_cell_segments.npy'))

        self.assertTrue(len(fibre_segments) > 0)
        self.assertTrue(len(cell_segments) > 0)

    def test_metric_analysis(self):

        with NamedTemporaryFile() as tmp_file:

            databases = self.image_analyser.metric_analysis(
                    self.multi_image, tmp_file.name,
                    self.fibre_networks, self.fibre_segments,
                    self.cell_segments)

            self.assertTrue(
                os.path.exists(tmp_file.name + '_global_metric.h5'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre_metric.h5'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_network_metric.h5'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_cell_metric.h5'))

        self.assertEqual(4, len(databases))

    def test_create_figures(self):
        pass
