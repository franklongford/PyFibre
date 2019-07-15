from unittest import mock, TestCase

from pyfibre.gui.options_pane import OptionsPane


class TestOptionsPane(TestCase):

    def setUp(self):

        self.options_pane = OptionsPane()

    def test___init__(self):

        self.assertFalse(self.options_pane.ow_metric)
        self.assertFalse(self.options_pane.ow_segment)
        self.assertFalse(self.options_pane.ow_network)

        self.assertEqual(1, self.options_pane.p_intensity[0])
        self.assertEqual(99, self.options_pane.p_intensity[1])

        #self.options_pane.configure_traits()