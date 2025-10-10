from unittest import TestCase

from pyfibre.tools.segmentation import rgb_segmentation
from pyfibre.testing.example_objects import generate_image
from pyfibre.testing.probe_objects import ProbeKmeansFilter


class TestSegmentation(TestCase):
    def setUp(self):
        (self.image, self.labels, self.binary, self.stack) = generate_image()
        self.bd_filter = ProbeKmeansFilter()

    def test_rgb_segmentation(self):
        stack = (self.image, self.image, self.image)

        fibre_mask, cell_mask = rgb_segmentation(stack, self.bd_filter)

        self.assertEqual((10, 10), fibre_mask.shape)
        self.assertEqual((10, 10), cell_mask.shape)
