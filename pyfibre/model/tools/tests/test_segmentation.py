from pyfibre.model.tools.segmentation import (
    rgb_segmentation
)
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_probe_graph)
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork
from pyfibre.tests.probe_classes.filters import ProbeBDFilter
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestSegmentation(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.fibre_networks = [ProbeFibreNetwork()]
        self.bd_filter = ProbeBDFilter()

    def test_rgb_segmentation(self):

        stack = (self.image,
                 self.image,
                 self.image)

        fibre_mask, cell_mask = rgb_segmentation(
            stack, self.bd_filter)

        self.assertEqual((10, 10), fibre_mask.shape)
        self.assertEqual((10, 10), cell_mask.shape)
