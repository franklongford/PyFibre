from unittest import TestCase

import numpy as np

from pyfibre.model.multi_image.multi_images import (
    SHGImage, SHGPLTransImage)


class TestSHGImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = SHGImage()

    def test_init_(self):

        self.assertEqual(1, len(self.multi_image))
        self.assertListEqual(
            [None],
            self.multi_image.image_stack
        )

    def test_assign_shg_image(self):

        self.multi_image.shg_image = self.image
        self.assertEqual(1, len(self.multi_image))
        self.assertEqual(
            id(self.image),
            id(self.multi_image.shg_image)
            )
        self.assertEqual(
            id(self.image),
            id(self.multi_image.image_stack[0])
        )


class TestPLSHGTransImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = SHGPLTransImage()

    def test_init_(self):

        self.assertEqual(3, len(self.multi_image))
        self.assertListEqual(
            [None, None, None],
            self.multi_image.image_stack
        )

    def test_assign_images(self):
        pl_image = np.zeros_like(self.image)

        self.multi_image.pl_image = pl_image
        self.assertEqual(
            id(pl_image),
            id(self.multi_image.pl_image)
        )
        self.assertEqual(
            id(pl_image),
            id(self.multi_image.image_stack[1])
        )

        trans_image = np.zeros_like(self.image)

        self.multi_image.trans_image = trans_image
        self.assertEqual(
            id(trans_image),
            id(self.multi_image.trans_image)
        )
        self.assertEqual(
            id(trans_image),
            id(self.multi_image.image_stack[2])
        )
