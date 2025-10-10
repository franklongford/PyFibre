import os
from tempfile import NamedTemporaryFile

import numpy as np
from skimage.measure import regionprops

from pyfibre.tests.probe_classes.utilities import generate_image
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase

from ..figures import (
    create_figure,
    create_hsb_image,
    create_tensor_image,
    create_region_image,
    create_network_image
)


class TestFigures(PyFibreTestCase):

    def setUp(self):

        (self.image, labels,
         self.binary, _) = generate_image()
        self.segments = regionprops(labels)
        self.fibre_network = ProbeFibreNetwork()

    def _check_pixels(self, image, channels, conditions):
        for channel in channels:
            for condition, expected in conditions:
                condition = f"self.image {condition}"
                with self.subTest(channel=channel,
                                  condition=condition,
                                  expected=expected):
                    indices = np.where(eval(condition))
                    self.assertTrue(
                        np.allclose(
                            image[..., channel][indices],
                            expected
                        )
                    )

    def test_create_figure(self):

        with NamedTemporaryFile() as temp_file:
            create_figure(self.image, temp_file.name)

            self.assertTrue(
                os.path.exists(temp_file.name + '.png')
            )

    def test_create_hsb_image(self):
        hsb_image = create_hsb_image(
            self.image, np.ones((10, 10)))

        self.assertEqual(
            (10, 10, 3), hsb_image.shape)
        self._check_pixels(
            hsb_image, [0], [(" >= 0", 1)])
        self._check_pixels(
            hsb_image, [1, 2], [(" >= 0", 0)])

    def test_create_tensor_image(self):
        tensor_image = create_tensor_image(self.image)

        self.assertEqual(
            (10, 10, 3), tensor_image.shape)
        self._check_pixels(
            tensor_image, [0, 1, 2], [(" == 0", 0)])
        self._check_pixels(
            tensor_image, [1], [(" == 10", 1)])
        self._check_pixels(
            tensor_image, [1, 2], [(" == 7", 0.7)])
        self._check_pixels(
            tensor_image, [1], [(" == 5", 0.5)])

    def test_create_segment_image(self):
        segment_image = create_region_image(
            self.image, self.segments
        )
        self.assertEqual(
            (10, 10, 3), segment_image.shape)
        self._check_pixels(
            segment_image, [0, 1, 2], [(" == 0", 0.0075)])
        self._check_pixels(
            segment_image, [1], [(" == 10", 0.75)])
        self._check_pixels(
            segment_image, [1, 2], [(" == 7", 0.52725)])
        self._check_pixels(
            segment_image, [1], [(" == 5", 0.37875)])

    def test_create_network_image(self):
        network_image = create_network_image(
            self.image,
            [self.fibre_network.graph])

        self.assertEqual(
            (10, 10, 3), network_image.shape)
        self._check_pixels(
            network_image, [1, 2], [(" == 0", 0)])
        self._check_pixels(
            network_image, [1], [(" == 10", 1)])
        self._check_pixels(
            network_image, [1, 2], [(" == 7", 0.7)])
        self._check_pixels(
            network_image, [1], [(" == 5", 0.5)])
