from unittest import TestCase

from traits.testing.unittest_tools import UnittestTools

from pyfibre.gui.metric_tab import MetricTab


class TestMetricTab(UnittestTools, TestCase):

    def setUp(self):

        self.metric_tab = MetricTab()

    def test___init__(self):

        self.assertIsNotNone(self.metric_tab.data)
        self.assertIsNotNone(self.metric_tab.plot)
        self.assertIsNotNone(self.metric_tab.tabular_adapter)

    def test(self):
        self.metric_tab.headers = ['A', 'B', 'C']
        self.metric_tab.data = [(0, 2, 4), ('h', 1, 2)]
        self.metric_tab.configure_traits()
