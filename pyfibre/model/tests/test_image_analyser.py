from unittest import TestCase, mock

import networkx as nx

from pyfibre.tests.probe_classes import ProbeSHGPLTransImage

from .. image_analyser import ImageAnalyser


LOAD_NETWORK_PATH = "networkx.read_gpickle"
SAVE_NETWORK_PATH = "networkx.write_gpickle"

LOAD_SEGMENT_PATH = "numpy.load"
SAVE_SEGMENT_PATH = "numpy.save"


def mock_load(*args, **kwargs):
    print('mock_load called')
    return True


class TestImageAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.image_analyser = ImageAnalyser()

    def test_get_ow_options(self):

        filename = "test_filename"

        ow_network, ow_segment, ow_metric = (
            self.image_analyser.get_analysis_options(filename))

        self.assertTrue(ow_network)
        self.assertTrue(ow_segment)
        self.assertTrue(ow_metric)

        with mock.patch(LOAD_NETWORK_PATH,
                        mock.mock_open(read_data=nx.Graph())):
            ow_network, ow_segment, ow_metric = (
                self.image_analyser.get_analysis_options(filename))

        self.assertFalse(ow_network)
        self.assertTrue(ow_segment)
        self.assertTrue(ow_metric)

    def test_network_analysis(self):

        with mock.patch(SAVE_NETWORK_PATH,
                        mock.mock_open()):
            network = self.image_analyser.network_analysis(
                self.multi_image, "test_filename")

        self.assertEqual(403, network.number_of_nodes())
        self.assertEqual(411, network.number_of_edges())
