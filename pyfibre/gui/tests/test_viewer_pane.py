from unittest import TestCase

from pyfibre.gui.viewer_pane import (
    ViewerPane
)
from pyfibre.tests.probe_classes import ProbeMultiImage


class TestViewerPane(TestCase):

    def setUp(self):

        self.multi_image = ProbeMultiImage()
        self.viewer_pane = ViewerPane()
