import os
from tempfile import NamedTemporaryFile
from unittest import TestCase, mock

from pyfibre.tests.probe_classes.multi_images import ProbeSHGPLTransImage
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork)

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


def mock_load_fibre_networks(*args, **kwargs):
    return [ProbeFibreNetwork()]


class TestImageAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.image_analyser = ImageAnalyser()

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

        self.assertEqual(655, network.number_of_nodes())
        self.assertEqual(688, network.number_of_edges())
        self.assertEqual(4, len(fibre_networks))

    def test_segment_analysis(self):

        with NamedTemporaryFile() as tmp_file:
            with mock.patch(
                    'pyfibre.model.image_analyser'
                    '.load_fibre_networks') as mock_loader:
                mock_loader.side_effect = mock_load_fibre_networks

                fibre_segments, cell_segments = (
                    self.image_analyser.segment_analysis(
                        self.multi_image, tmp_file.name))

            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre_segments.npy'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_cell_segments.npy'))

        self.assertEqual(13, len(fibre_segments))
        self.assertEqual(1, len(cell_segments))

    def test_metric_analysis(self):
        pass

    def test_create_figures(self):
        pass
