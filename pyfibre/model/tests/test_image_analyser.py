import os
from tempfile import NamedTemporaryFile
from unittest import TestCase, mock

import networkx as nx
import numpy as np

from pyfibre.tests.probe_classes import ProbeSHGPLTransImage

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

    def test_get_ow_options(self):

        filename = "test_filename"
        return_json = [{'fibre_networks': []}]

        ow_network, ow_segment, ow_metric = (
            self.image_analyser.get_analysis_options(
                self.multi_image, filename))

        self.assertTrue(ow_network)
        self.assertTrue(ow_segment)
        self.assertTrue(ow_metric)

        with mock.patch(
                LOAD_NETWORK_PATH,
                mock.mock_open(read_data=nx.Graph())), \
                mock.patch(
                    LOAD_JSON_PATH,
                    mock.MagicMock(side_effect=return_json)):

            ow_network, ow_segment, ow_metric = (
                self.image_analyser.get_analysis_options(
                    self.multi_image, filename))

            self.assertFalse(ow_network)
            self.assertTrue(ow_segment)
            self.assertTrue(ow_metric)

        with mock.patch(
                LOAD_NETWORK_PATH,
                mock.mock_open(read_data=nx.Graph())), \
                mock.patch(
                    LOAD_JSON_PATH,
                    mock.MagicMock(side_effect=return_json)), \
                mock.patch(
                    LOAD_REGION_PATH,
                    mock.mock_open(read_data=np.ones((10, 10)))):
            ow_network, ow_segment, ow_metric = (
                self.image_analyser.get_analysis_options(
                    self.multi_image, filename))

            self.assertFalse(ow_network)
            self.assertFalse(ow_segment)
            self.assertTrue(ow_metric)

    def test_network_analysis(self):

        with NamedTemporaryFile() as tmp_file:

            network, fibre_networks = (
                self.image_analyser.network_analysis(
                    self.multi_image, tmp_file.name))

            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre_networks.json'))

        self.assertEqual(403, network.number_of_nodes())
        self.assertEqual(411, network.number_of_edges())

        self.assertEqual(10, len(fibre_networks))

    def test_segment_analysis(self):
        pass
