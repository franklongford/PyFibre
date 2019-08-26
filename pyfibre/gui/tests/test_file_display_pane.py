import os
from unittest import TestCase

from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.io.tif_reader import TIFReader

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestFileDisplayPane(TestCase):

    def setUp(self):

        self.file_display = FileDisplayPane()
        self.file_path = (
            pyfibre_dir + '/tests/fixtures/test-pyfibre-pl-shg-Stack.tif'
        )

    def test___init__(self):
        self.assertIsInstance(self.file_display.tif_reader, TIFReader)

    def test_add_file(self):

        self.file_display.add_files(self.file_path)

        table_row = self.file_display.file_table[0]
        self.assertEqual(
            pyfibre_dir + '/tests/fixtures/test-pyfibre',
            table_row.name)
        self.assertTrue(table_row.shg)
        self.assertTrue(table_row.pl)

    def test_add_directory(self):

        temp_dir = pyfibre_dir + '/tests/fixtures'
        self.file_display.add_files(temp_dir)

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
        self.assertEqual(0, len(self.file_display.tif_reader.files))

        self.file_display.add_files(self.file_path)

        self.assertEqual(1, len(self.file_display.file_table))
        self.assertEqual(1, len(self.file_display.tif_reader.files))

    def test_filter_files(self):

        self.file_display.add_files(self.file_path)
        self.file_display.filter_files('pyfibre')

        self.assertEqual(1, len(self.file_display.file_table))
        self.assertEqual(1, len(self.file_display.tif_reader.files))

        self.file_display.filter_files('sci-pyfibre')

        self.assertEqual(0, len(self.file_display.file_table))
        self.assertEqual(0, len(self.file_display.tif_reader.files))

    def test_shg_pl_requirements(self):

        shg_file = (
                pyfibre_dir +
                '/tests/fixtures/test-pyfibre-shg-Stack.tif'
        )
        self.file_display.tif_reader.pl = False
        self.file_display.add_files(shg_file)

        table_row = self.file_display.file_table[0]
        self.assertTrue(table_row.shg)
        self.assertFalse(table_row.pl)
