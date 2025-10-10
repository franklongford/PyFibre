from chaco.api import ArrayPlotData
from traits.testing.unittest_tools import UnittestTools

from pyfibre.tests.probe_classes.multi_images import ProbeMultiImage
from pyfibre.tests.probe_classes.viewers import ProbeMultiImageViewer
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestBaseMultiImageViewer(UnittestTools, PyFibreTestCase):

    def setUp(self):
        self.image = ProbeMultiImage()
        self.viewer = ProbeMultiImageViewer(
            multi_image=self.image
        )
        self.display_tab = self.viewer.display_tabs[0]

    def test_init(self):
        self.assertEqual(1, len(self.viewer.display_tabs))

    def test_plot_data(self):
        plot_data = ArrayPlotData({'test': self.image})

        with self.assertTraitChanges(self.display_tab, "plot_data"):
            self.display_tab.plot_data = plot_data

    def test_update_viewer(self):
        new_image = ProbeMultiImage()

        with self.assertTraitChanges(self.display_tab, "updated"):
            self.viewer.update_viewer(new_image)

        self.assertEqual(new_image, self.viewer.multi_image)
