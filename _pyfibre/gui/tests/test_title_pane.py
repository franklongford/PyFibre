from unittest import TestCase

from pyfibre.gui.title_pane import TitlePane


class TestTitlePane(TestCase):

    def setUp(self):

        self.title_pane = TitlePane()

    def test___init__(self):
        self.assertEqual(
            'pyfibre.title_pane',
            self.title_pane.id
        )
        self.assertEqual(
            'Title Pane',
            self.title_pane.name
        )
