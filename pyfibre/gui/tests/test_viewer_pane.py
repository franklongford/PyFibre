from unittest import TestCase

import numpy as np

from pyfibre.gui.viewer_pane import (
    ViewerPane
)
from pyfibre.gui.image_tab import ImageTab
from pyfibre.model.multi_image.multi_image import SHGPLTransImage


class TestImageTab(TestCase):

    def setUp(self):

        self.multi_image = SHGPLTransImage()
        self.multi_image.assign_shg_image(np.ones((50, 50)))
        self.image_tab = ImageTab(
            label='SHG',
            image=self.multi_image.shg_image
        )

    def test___init__(self):

        self.assertEqual('SHG', self.image_tab.label)
        self.assertIsNotNone(self.multi_image.shg_image)
        self.assertIsNotNone(self.image_tab.image)
        self.assertIsNotNone(self.image_tab.plot)


class TestViewerPane(TestCase):

    def setUp(self):

        self.viewer_pane = ViewerPane()
