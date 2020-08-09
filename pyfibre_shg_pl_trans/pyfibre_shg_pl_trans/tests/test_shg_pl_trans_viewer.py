from unittest import TestCase

from pyfibre_shg_pl_trans.shg_pl_trans_viewer import SHGPLTransViewer

from .probe_classes import ProbeSHGPLTransImage


class TestSHGPLTransViewer(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.viewer = SHGPLTransViewer()

    def test_update_viewer(self):

        self.viewer.update_viewer(self.multi_image)

        self.assertEqual(
            self.multi_image,
            self.viewer.multi_image)
        self.assertListEqual(
            ['SHG', 'PL', 'Trans'],
            self.viewer.selected_tab.image_labels
        )

        self.assertListEqual(
            [], self.viewer.network_tab.networks)
        self.assertListEqual(
            [], self.viewer.fibre_tab.networks)
        self.assertListEqual(
            [], self.viewer.fibre_segment_tab.segments)
