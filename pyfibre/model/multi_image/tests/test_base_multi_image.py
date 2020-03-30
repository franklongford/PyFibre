from unittest import TestCase

import numpy as np

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage


class TestBaseMultiImage(TestCase):

    def setUp(self):

        self.multi_image = BaseMultiImage()

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

    def test___init__(self):

        self.assertEqual(0, len(self.multi_image))
        self.assertIsNone(self.multi_image.shape)
        self.assertIsNone(self.multi_image.size)
        self.assertIsNone(self.multi_image.ndim)
        self.assertListEqual([], self.multi_image.image_stack)

    def test_append_remove(self):

        self.multi_image.append(self.image)

        self.assertEqual(1, len(self.multi_image))
        self.assertEqual((15, 15), self.multi_image.shape)
        self.assertEqual(15**2, self.multi_image.size)
        self.assertEqual(2, self.multi_image.ndim)

        with self.assertRaises(ValueError):
            self.multi_image.append(np.ones((10, 10)))

        self.multi_image.remove(self.image)

        self.assertEqual(0, len(self.multi_image))
        self.assertIsNone(self.multi_image.shape)
        self.assertIsNone(self.multi_image.size)

    def test_preprocess_images(self):

        with self.assertRaises(NotImplementedError):
            self.multi_image.preprocess_images()
