from unittest import TestCase

import numpy as np

from pyfibre.model.multi_image.shg_image import SHGImage


class TestSHGImage(TestCase):

    def setUp(self):

        self.image = np.ones((15, 15))
        self.image[5: 5] = 0
        self.image[0: 0] = 2

        self.multi_image = SHGImage(
            name='test-shg',
            path='/path/to/image'
        )

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
