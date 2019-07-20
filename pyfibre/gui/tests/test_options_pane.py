from unittest import mock, TestCase

from pyfibre.gui.options_pane import OptionsPane


class TestOptionsPane(TestCase):

    def setUp(self):

        self.options_pane = OptionsPane()

    def test___init__(self):

        self.assertFalse(self.options_pane.ow_figure)
        self.assertFalse(self.options_pane.ow_metric)
        self.assertFalse(self.options_pane.ow_segment)
        self.assertFalse(self.options_pane.ow_network)

        self.assertEqual(5, self.options_pane.n_denoise)
        self.assertEqual(35, self.options_pane.m_denoise)

        self.assertEqual(1, self.options_pane.low_intensity)
        self.assertEqual(99, self.options_pane.high_intensity)

        #self.options_pane.configure_traits()