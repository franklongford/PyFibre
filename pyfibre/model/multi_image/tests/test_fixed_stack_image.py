from unittest import TestCase

import numpy as np

from pyfibre.tests.probe_classes import ProbeFixedStackImage


class TestFixedStackImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = ProbeFixedStackImage()

    def test_default_stack(self):

        self.assertListEqual(
            [None], self.multi_image.image_stack
        )

    def test_verify_stack(self):
        self.assertTrue(
            self.multi_image.verify_stack([self.image]))

        self.assertFalse(
            self.multi_image.verify_stack(
                [self.image, self.image]))
        self.assertFalse(
            self.multi_image.verify_stack(
                [self.image, np.ones((10, 10))]))
        self.assertFalse(
            self.multi_image.verify_stack(
                [np.ones((15, 15, 3))]))

    def test_preprocess_images(self):

        self.multi_image.p_intensity = (10, 90)
        self.multi_image._stack_len = 1
        self.multi_image.image_stack[0] = self.image

        self.multi_image.preprocess_images()
        self.assertEqual(
            np.ones(1),
            np.unique(self.multi_image.image_stack[0]))
