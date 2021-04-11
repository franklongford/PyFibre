from unittest import TestCase
import pandas as pd

from traits.testing.unittest_tools import UnittestTools

from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.tests.probe_classes.multi_images import (
    ProbeMultiImage)


class TestSegmentImageTab(UnittestTools, TestCase):

    def setUp(self):
        self.multi_image = ProbeMultiImage()
        example_data = pd.DataFrame(
            {'A': ['l', 'h'], 'B': [2, 1], 'C': [4, 2]}
        )
        self.metric_tab = SegmentImageTab(
            multi_image=self.multi_image,
            data=example_data
        )

    def test___init__(self):
        self.assertIsNotNone(self.metric_tab.data)
        self.assertIsNotNone(self.metric_tab.plot)
        self.assertIsNotNone(self.metric_tab.image_plot)
        self.assertIsNotNone(self.metric_tab.tabular_adapter)

    def test_init_data(self):
        example_data = pd.DataFrame(
            {'A': ['l', 'h'], 'B': [2, 1], 'C': [4, 2]}
        )
        self.metric_tab.data = example_data

        self.assertListEqual(
            ['', 'A', 'B', 'C'],
            self.metric_tab.headers
        )
        self.assertListEqual(
            ['B', 'C'],
            self.metric_tab._display_cols
        )
