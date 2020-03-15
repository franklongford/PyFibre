from unittest import TestCase

from skimage.io import imread

from pyfibre.model.tools.segmentation import (
    rgb_segmentation
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)
from pyfibre.tests.fixtures import test_image_path


class TestSegmentation(TestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.image_stack = imread(test_image_path).mean(axis=-1)
        for image in self.image_stack:
            image /= image.max()

    def test_cell_segmentation(self):

        stack = (self.image_stack[0],
                 self.image_stack[1],
                 self.image_stack[2])

        fibre_mask, cell_mask = rgb_segmentation(stack)

    def test_fibre_segmentation(self):
        pass
