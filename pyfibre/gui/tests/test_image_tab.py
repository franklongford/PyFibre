from unittest import TestCase

from traits.testing.unittest_tools import UnittestTools

from pyfibre.gui.image_tab import ImageTab
from pyfibre.tests.probe_classes import (
    ProbeImageTab, ProbeNetworkImageTab, ProbeSegmentImageTab,
    ProbeMultiImage)


class TestImageTab(UnittestTools, TestCase):

    def setUp(self):

        self.image_tab = ProbeImageTab()
        self.multi_image = ProbeMultiImage()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.multi_image)
        self.assertIsNotNone(self.image_tab.plot)
        self.assertEqual('Test 1', self.image_tab.selected_label)

    def test_empty_image_dict(self):

        image_tab = ImageTab()

        self.assertEqual('', image_tab.selected_label)
        self.assertDictEqual({}, image_tab._image_dict)
        self.assertDictEqual({}, image_tab.plot.data.arrays)
        self.assertListEqual([], image_tab.image_labels)

    def test_image_labels(self):

        self.assertListEqual(
            ['Test 1', 'Test 2'], self.image_tab.image_labels)
        with self.assertTraitChanges(self.image_tab, 'plot'):
            self.image_tab.selected_label = 'Test 2'


class TestNetworkImageTab(TestCase):

    def setUp(self):

        self.image_tab = ProbeNetworkImageTab()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.networks)


class TestSegmentImageTab(TestCase):

    def setUp(self):

        self.image_tab = ProbeSegmentImageTab()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.segments)
        self.assertEqual(
            len(self.image_tab.segments),
            len(self.image_tab.regions)
        )
