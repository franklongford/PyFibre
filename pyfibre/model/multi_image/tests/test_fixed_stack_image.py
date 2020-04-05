from unittest import TestCase

import numpy as np

from pyfibre.model.multi_image.fixed_stack_image import (
    FixedStackImage)


class TestFixedStackImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = FixedStackImage()

    def test_default_stack(self):

        self.multi_image._max_len = 3
        self.assertListEqual(
            [None, None, None], self.multi_image.image_stack
        )

    def test_preprocess_images(self):

        self.multi_image.p_intensity = (10, 90)
        self.multi_image._max_len = 1
        self.multi_image.image_stack[0] = self.image

        self.multi_image.preprocess_images()
        self.assertEqual(
            np.ones(1),
            np.unique(self.multi_image.image_stack[0]))
