from unittest import TestCase

from traits.testing.unittest_tools import UnittestTools

from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.tests.probe_classes.gui_objects import ProbeTableRow
from pyfibre.tests.probe_classes.multi_images import ProbeMultiImage
from pyfibre.tests.probe_classes.viewers import ProbeMultiImageViewer


class TestViewerPane(UnittestTools, TestCase):

    def setUp(self):

        self.table_row = ProbeTableRow()
        self.multi_image = ProbeMultiImage()
        self.viewer = ProbeMultiImageViewer()
        self.viewer_pane = ViewerPane(
            supported_viewers={ProbeMultiImage: self.viewer}
        )

    def test_update_viewer(self):

        self.assertIsNone(self.viewer_pane.selected_image)
        self.viewer_pane.selected_image = self.multi_image

        self.assertEqual(
            self.viewer, self.viewer_pane.selected_viewer)
        self.assertEqual(
            self.viewer_pane.selected_image,
            self.viewer.multi_image)

    def test_update(self):
        display_tab = self.viewer.display_tabs[0]
        self.viewer_pane.selected_image = self.multi_image

        with self.assertTraitChanges(display_tab, "updated"):
            self.viewer_pane.update()
