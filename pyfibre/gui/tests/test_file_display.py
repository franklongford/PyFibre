from unittest import TestCase

from tkinter import Tk
from pyfibre.gui.tk.file_display import FileDisplay


class TestFileDisplay(TestCase):

    def setUp(self):

        root = Tk()
        self.file_display = FileDisplay(root, '')


    def test___init__(self):

        self.assertIsNone(self.file_display.select_im_button)
        self.assertIsInstance(self.file_display.select_im_button)