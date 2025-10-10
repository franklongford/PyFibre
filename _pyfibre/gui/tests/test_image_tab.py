from unittest import TestCase
import numpy as np

from traits.testing.unittest_tools import UnittestTools

from pyfibre.gui.image_tab import ImageTab
from pyfibre.tests.probe_classes.gui_objects import (
    ProbeImageTab, ProbeNetworkImageTab, ProbeSegmentImageTab)
from pyfibre.tests.probe_classes.multi_images import (
    ProbeMultiImage)
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, ProbeSegment)


class TestImageTab(UnittestTools, TestCase):

    def setUp(self):

        self.image_tab = ProbeImageTab()
        self.multi_image = ProbeMultiImage()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.multi_image)
        self.assertIsNotNone(self.image_tab.image_plot)
        self.assertEqual('Test 0', self.image_tab.selected_label)

    def test_empty_image_dict(self):

        image_tab = ImageTab()

        self.assertIsNone(image_tab.selected_label)
        self.assertIsNone(image_tab.image_plot.data)
        self.assertListEqual([], image_tab.image_labels)

    def test_image_labels(self):
        self.image_tab.update_tab()
        self.assertListEqual(
            ['Test 0', 'Test 1'], self.image_tab.image_labels)
        with self.assertTraitChanges(self.image_tab, 'image_plot'):
            self.image_tab.selected_label = 'Test 1'

    def test_multi_image_change(self):
        image_tab = ImageTab()

        with self.assertTraitChanges(image_tab, 'image_plot'):
            image_tab.multi_image = self.multi_image
            image_tab.update_tab()

    def test_brightness_change(self):
        self.image_tab.update_tab()
        orig_image = self.multi_image.image_dict[self.image_tab.selected_label]

        np.testing.assert_array_almost_equal(
            0.1 * orig_image,
            self.image_tab.image_plot.data[self.image_tab.selected_label],
            2
        )


class TestNetworkImageTab(UnittestTools, TestCase):

    def setUp(self):

        self.image_tab = ProbeNetworkImageTab()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.networks)

    def test_image_data(self):
        new_networks = [ProbeFibreNetwork().graph]

        with self.assertTraitChanges(self.image_tab, "image_data"):
            self.image_tab.networks = new_networks
            self.image_tab.update_tab()


class TestSegmentImageTab(UnittestTools, TestCase):

    def setUp(self):

        self.image_tab = ProbeSegmentImageTab()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.segments)

    def test_image_data(self):
        new_segments = [ProbeSegment()]

        with self.assertTraitChanges(self.image_tab, "image_data"):
            self.image_tab.segments = new_segments
            self.image_tab.update_tab()
