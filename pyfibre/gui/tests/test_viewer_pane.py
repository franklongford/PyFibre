from unittest import TestCase

from pyfibre.gui.viewer_pane import (
    ViewerPane
)
from pyfibre.tests.probe_classes.gui_objects import ProbeTableRow
from pyfibre.shg_pl_trans.tests.probe_classes import ProbeSHGPLTransImage


class TestViewerPane(TestCase):

    def setUp(self):

        self.table_row = ProbeTableRow()
        self.multi_image = ProbeSHGPLTransImage()
        self.viewer_pane = ViewerPane()

    def test_update_viewer(self):

        self.assertIsNone(self.viewer_pane.selected_image)

        self.viewer_pane.update_viewer(
            self.multi_image, 'some/path/with/no/analysis')

        self.assertIsNotNone(self.viewer_pane.selected_image)
        self.assertEqual(
            self.viewer_pane.selected_image,
            self.viewer_pane.selected_tab.multi_image)
        self.assertListEqual(
            ['SHG', 'PL', 'Trans'],
            self.viewer_pane.selected_tab.image_labels
        )

        self.assertListEqual(
            [], self.viewer_pane.network_tab.networks)
        self.assertListEqual(
            [], self.viewer_pane.fibre_tab.networks)
        self.assertListEqual(
            [], self.viewer_pane.fibre_segment_tab.segments)

    def test_selected_tab(self):

        self.assertEqual(
            self.viewer_pane.selected_tab,
            self.viewer_pane.multi_image_tab)

        self.viewer_pane.selected_image = self.multi_image
        self.viewer_pane.selected_tab.selected_label = 'PL'
        self.viewer_pane.selected_tab = self.viewer_pane.image_tab_list[1]

        self.assertEqual(
            'PL', self.viewer_pane.selected_tab.selected_label
        )
