import os
from unittest import TestCase

from pyfibre.gui.file_display_pane import FileDisplayPane

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestFileDisplayPane(TestCase):

    def setUp(self):

        self.file_display = FileDisplayPane()
        self.file_path = (
            pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif'
        )

    def test___init__(self):
        self.assertEqual(0, len(self.file_display.input_files))
        self.assertEqual(0, len(self.file_display.input_prefixes))

        #self.file_display.configure_traits()

    def test_add_file(self):

        self.file_display.add_files(self.file_path)

        self.assertEqual(
            self.file_path,
            self.file_display.input_files[0])

        self.assertEqual(
            pyfibre_dir + '/tests/stubs/test-pyfibre',
            self.file_display.input_prefixes[0])

        table_row = self.file_display.file_table[0]
        self.assertEqual(
            pyfibre_dir + '/tests/stubs/test-pyfibre',
            table_row.name)
        self.assertTrue(table_row.shg)
        self.assertTrue(table_row.pl)

    def test_add_directory(self):

        temp_dir = pyfibre_dir + '/tests/stubs'
        self.file_display.add_files(temp_dir)

        self.assertEqual(
            temp_dir + '/test-pyfibre-pl-shg-Stack.tif',
            self.file_display.input_files[0])

        self.assertEqual(
            pyfibre_dir + '/tests/stubs/test-pyfibre',
            self.file_display.input_prefixes[0])

        table_row = self.file_display.file_table[0]
        self.assertEqual(
            temp_dir + '/test-pyfibre',
            table_row.name)
        self.assertTrue(table_row.shg)
        self.assertTrue(table_row.pl)

    def test_remove_file(self):

        self.file_display.add_files(self.file_path)
        self.file_display.remove_file([self.file_display.file_table[0]])

        self.assertEqual(0, len(self.file_display.file_table))