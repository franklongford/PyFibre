from unittest import TestCase

from pyfibre.gui.image_tab import ImageTab
from pyfibre.tests.probe_classes import (
    ProbeImageTab, ProbeNetworkImageTab, ProbeSegmentImageTab)


class TestImageTab(TestCase):

    def setUp(self):

        self.image_tab = ProbeImageTab()

    def test___init__(self):

        self.assertIsNotNone(self.image_tab.multi_image)
        self.assertIsNotNone(self.image_tab.plot)
        self.assertEqual('Test', self.image_tab.selected_label)

    def test_empty_image_dict(self):

        image_tab = ImageTab()

        self.assertEqual('', image_tab.selected_label)
        self.assertDictEqual({}, image_tab._image_dict)
        self.assertDictEqual({}, image_tab.plot.data.arrays)


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
