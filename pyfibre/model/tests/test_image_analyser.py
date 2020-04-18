import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

from pyfibre.tests.probe_classes.multi_images import ProbeSHGPLTransImage

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

        self.assertEqual(403, network.number_of_nodes())
        self.assertEqual(411, network.number_of_edges())

        self.assertEqual(10, len(fibre_networks))

    def test_segment_analysis(self):
        pass
