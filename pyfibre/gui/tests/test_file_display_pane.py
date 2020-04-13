import os
from unittest import TestCase

from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.tests.fixtures import (
    directory, test_shg_pl_trans_image_path, test_shg_image_path)

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestFileDisplayPane(TestCase):

    def setUp(self):

        self.file_display = FileDisplayPane(
            supported_readers=['SHG-PL-Trans']
        )
        self.file_path = test_shg_pl_trans_image_path

    def test_add_file(self):

        self.file_display.add_files(self.file_path)
        self.assertEqual(1, len(self.file_display.file_table))

        table_row = self.file_display.file_table[0]
        self.assertEqual(
            directory + '/test-pyfibre',
            table_row.name)
        self.assertTrue(table_row.shg)
        self.assertTrue(table_row.pl)

        self.file_display.add_files(test_shg_image_path)
        self.assertEqual(1, len(self.file_display.file_table))

    def test_add_directory(self):

        self.file_display.add_files(directory)
        self.assertEqual(1, len(self.file_display.file_table))

        table_row = self.file_display.file_table[0]
        self.assertEqual(
            directory + '/test-pyfibre',
            table_row.name)
        self.assertTrue(table_row.shg)
        self.assertTrue(table_row.pl)

    def test_remove_file(self):

        self.file_display.add_files(self.file_path)
        self.file_display.remove_file(
            [self.file_display.file_table[0]])

        self.assertEqual(0, len(self.file_display.file_table))

        self.file_display.add_files(self.file_path)

        self.assertEqual(1, len(self.file_display.file_table))

    def test_filter_files(self):

        self.file_display.add_files(self.file_path)
        self.file_display.filter_files('pyfibre')

        self.assertEqual(1, len(self.file_display.file_table))

        self.file_display.filter_files('sci-pyfibre')

        self.assertEqual(0, len(self.file_display.file_table))
