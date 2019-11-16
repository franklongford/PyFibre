from unittest import TestCase
import numpy as np

from pyfibre.io.multi_image import MultiLayerImage


class TestMultiLayerImage(TestCase):

    def setUp(self):

        self.multi_image = MultiLayerImage()

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

    def test___init__(self):

        self.assertEqual(0, len(self.multi_image))
        self.assertIsNone(self.multi_image.shape)
        self.assertIsNone(self.multi_image.size)

        self.assertFalse(self.multi_image.image_stack)

    def test_append_remove(self):

        self.multi_image.append(self.image)

        self.assertEqual(1, len(self.multi_image))
        self.assertEqual((15, 15), self.multi_image.shape)
        self.assertEqual(15**2, self.multi_image.size)

        with self.assertRaises(ValueError):
            self.multi_image.append(np.ones((10, 10)))

        self.multi_image.remove(self.image)

        self.assertEqual(0, len(self.multi_image))
        self.assertIsNone(self.multi_image.shape)
        self.assertIsNone(self.multi_image.size)

    def test_preprocess_images(self):

        self.multi_image.p_intensity = (10, 90)

        self.multi_image.append(self.image)
        self.multi_image.preprocess_images()

        self.assertEqual(
            np.ones(1),
            np.unique(self.multi_image.image_stack[0]))
