from unittest import TestCase

import numpy as np

from pyfibre.model.objects.multi_image import (
    MultiImage, SHGPLImage, SHGPLTransImage)


class TestMultiImage(TestCase):

    def setUp(self):

        self.multi_image = MultiImage()

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


class TestPLSHGImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = SHGPLImage()

    def test_init_(self):

        self.assertListEqual(
            [None, None],
            self.multi_image.image_stack
        )

    def test_assign_images(self):

        self.multi_image.assign_shg_image(self.image)
        self.assertEqual(
            id(self.image),
            id(self.multi_image.shg_image)
            )
        self.assertEqual(
            id(self.image),
            id(self.multi_image.image_stack[0])
        )

        pl_image = np.zeros_like(self.image)

        self.multi_image.assign_pl_image(pl_image)
        self.assertEqual(
            id(pl_image),
            id(self.multi_image.pl_image)
        )
        self.assertEqual(
            id(pl_image),
            id(self.multi_image.image_stack[1])
        )


class TestPLSHGTransImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = SHGPLTransImage()

    def test_init_(self):

        self.assertListEqual(
            [None, None, None],
            self.multi_image.image_stack
        )

    def test_assign_images(self):

        trans_image = np.zeros_like(self.image)

        self.multi_image.assign_trans_image(trans_image)
        self.assertEqual(
            id(trans_image),
            id(self.multi_image.trans_image)
        )
        self.assertEqual(
            id(trans_image),
            id(self.multi_image.image_stack[2])
        )
