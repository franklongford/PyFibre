import numpy as np

from skimage.io import imread

from pyfibre.model.tools.segmentation import (
    rgb_segmentation, create_composite_rgb_image
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)
from pyfibre.tests.fixtures import test_image_path
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestSegmentation(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.image_stack = imread(test_image_path).mean(axis=-1)
        for image in self.image_stack:
            image /= image.max()

    def test_cell_segmentation(self):

        cell_segments, fibre_segments = rgb_segmentation(
            self.image_stack[0],
            self.image_stack[1],
            self.image_stack[2]
        )

    def test_fibre_segmentation(self):
        pass

    def test_create_composite_rgb_image(self):

        composite_image_stack = create_composite_rgb_image(
            *self.image_stack
        )

        self.assertEqual(
            (200, 200, 3),
            composite_image_stack.shape)
        self.assertArrayAlmostEqual(
            np.ones((200, 200)),
            (composite_image_stack ** 2).sum(axis=-1)
        )
