from unittest import TestCase

from pyfibre.gui.viewer_pane import (
    ViewerPane
)
from pyfibre.tests.probe_classes import ProbeTableRow


class TestViewerPane(TestCase):

    def setUp(self):

        self.table_row = ProbeTableRow()
        self.viewer_pane = ViewerPane()

    def test_open_file(self):

        self.assertIsNone(self.viewer_pane.selected_image)
        self.viewer_pane.selected_row = self.table_row

        self.assertIsNotNone(self.viewer_pane.selected_image)
        self.assertEqual(
            self.viewer_pane.selected_image,
            self.viewer_pane.selected_tab.multi_image)
        self.assertListEqual(
            ['SHG', 'PL', 'Trans'],
            self.viewer_pane.selected_tab.image_labels
        )

    def test_selected_tab(self):

        self.assertEqual(
            self.viewer_pane.selected_tab,
            self.viewer_pane.multi_image_tab)

        self.viewer_pane.selected_row = self.table_row
        self.viewer_pane.selected_tab.selected_label = 'PL'
        self.viewer_pane.selected_tab = self.viewer_pane.image_tab_list[1]

        self.assertEqual(
            'PL', self.viewer_pane.selected_tab.selected_label
        )
