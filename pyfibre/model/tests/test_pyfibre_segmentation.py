from unittest import TestCase

from pyfibre.tests.probe_classes import ProbeSHGPLTransImage

from ..pyfibre_segmentation import cell_segmentation


class TestPyFibreSegmentation(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()

    def test_cell_segmentation(self):
        pass

