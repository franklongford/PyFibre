from unittest import TestCase
import numpy as np

from pyfibre.io.multi_image import MultiLayerImage


class TestMultiLayerImage(TestCase):

    def setUp(self):

        self.multi_image = MultiLayerImage(
            shg_analysis=True)

    def test___init__(self):

        self.assertFalse(self.multi_image.ow_segment)
        self.assertFalse(self.multi_image.ow_metric)
        self.assertFalse(self.multi_image.ow_figure)
        self.assertFalse(self.multi_image.ow_network)

        self.assertIsNone(self.multi_image.image_shg)
        self.assertIsNone(self.multi_image.image_pl)
        self.assertIsNone(self.multi_image.image_tran)

    def test_assign_image(self):

        self.multi_image.image_shg = np.ones((50, 50))
        self.multi_image.preprocess_image_shg()

        self.assertTrue(self.multi_image.shg_analysis)