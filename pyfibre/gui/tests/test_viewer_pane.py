from unittest import TestCase

import numpy as np

from pyfibre.gui.viewer_pane import (
    ViewerPane, ImageTab
)
from pyfibre.io.multi_image import MultiLayerImage


class TestImageTab(TestCase):

    def setUp(self):

        self.multi_image = MultiLayerImage()
        self.image_tab = ImageTab(
            label='SHG',
            image=self.multi_image.image_shg
        )

    def test___init__(self):

        self.assertEqual('SHG', self.image_tab.label)
        self.assertIsNone(self.image_tab.image)
        self.assertIsNone(self.image_tab.plot)

    def test_assign_image(self):

        self.multi_image.trait_set(image_shg=np.ones((50, 50)))

        self.assertIsNotNone(self.multi_image.image_shg)

        print(self.multi_image.image_shg)

        image_tab = ImageTab(
            label='SHG',
            image=self.multi_image.image_shg
        )

        self.assertIsNotNone(image_tab.image)
