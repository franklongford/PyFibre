from unittest import TestCase
import pandas as pd

from traits.testing.unittest_tools import UnittestTools

from pyfibre.tests.probe_classes.gui_objects import (
    ProbeImageMetricTab)


class TestMetricTab(UnittestTools, TestCase):

    def setUp(self):
        self.example_data = pd.DataFrame(
            {'A': ['l', 'h'], 'B': [2, 1], 'C': [4, 2]}
        )
        self.metric_tab = ProbeImageMetricTab(
            data=self.example_data
        )

    def test___init__(self):

        self.assertIsNotNone(self.metric_tab.data)
        self.assertIsNotNone(self.metric_tab.plot)
        self.assertIsNotNone(self.metric_tab.tabular_adapter)

    def test_init_data(self):
        self.assertListEqual(
            ['', 'A', 'B', 'C'],
            self.metric_tab.headers
        )
        self.assertListEqual(
            ['B', 'C'],
            self.metric_tab._display_cols
        )
