from unittest import TestCase

import numpy as np

from pyfibre.tests.probe_classes.multi_images import ProbeMultiImage


class TestBaseMultiImage(TestCase):

    def setUp(self):

        self.multi_image = ProbeMultiImage()
        self.image = np.ones((10, 10))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

    def test___init__(self):

        self.assertEqual(2, len(self.multi_image))
        self.assertIsNotNone(self.multi_image.shape)
        self.assertIsNotNone(self.multi_image.size)
        self.assertIsNotNone(self.multi_image.ndim)

    def test_append_remove(self):

        self.multi_image.append(self.image)

        self.assertEqual(3, len(self.multi_image))
        self.assertEqual((10, 10), self.multi_image.shape)
        self.assertEqual(100, self.multi_image.size)
        self.assertEqual(2, self.multi_image.ndim)

        with self.assertRaises(ValueError):
            self.multi_image.append(np.ones((15, 15)))

        self.multi_image.remove(self.image)
        self.assertEqual(2, len(self.multi_image))

        with self.assertRaises(IndexError):
            self.multi_image.remove(self.image)

    def test_to_array(self):
        self.multi_image.image_stack = [self.image]
        image_array = self.multi_image.to_array()
        self.assertEqual((1, 10, 10), image_array.shape)
