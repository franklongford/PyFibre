import os
from unittest import TestCase

from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.tests.fixtures import (
    directory,
    test_image_path)
from pyfibre.tests.probe_classes.readers import ProbeMultiImageReader

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestFileDisplayPane(TestCase):

    def setUp(self):

        self.file_display = FileDisplayPane(
            supported_readers={'Probe': ProbeMultiImageReader()}
        )
        self.file_path = test_image_path

    def test_add_file(self):

        self.file_display.add_files(self.file_path)
        self.assertEqual(1, len(self.file_display.file_table))

        table_row = self.file_display.file_table[0]
        self.assertEqual('probe-file', table_row.name)
        self.assertEqual('Probe', table_row.tag)
        self.assertListEqual(
            [test_image_path],
            table_row.file_names)

        self.file_display.add_files(test_image_path)
        self.assertEqual(1, len(self.file_display.file_table))

    def test_add_directory(self):

        self.file_display.add_files(directory)
        self.assertEqual(1, len(self.file_display.file_table))

        table_row = self.file_display.file_table[0]
        self.assertEqual('probe-file', table_row.name)
        self.assertEqual('Probe', table_row.tag)
        self.assertListEqual(
            [test_image_path],
            table_row.file_names)

    def test_remove_file(self):

        self.file_display.add_files(self.file_path)
        self.file_display.remove_file(
            [self.file_display.file_table[0]])

        self.assertEqual(0, len(self.file_display.file_table))

        self.file_display.add_files(self.file_path)

        self.assertEqual(1, len(self.file_display.file_table))

    def test_filter_files(self):

        self.file_display.add_files(self.file_path)
        self.file_display.filter_files('probe-file')

        self.assertEqual(1, len(self.file_display.file_table))

        self.file_display.filter_files('sci-pyfibre')

        self.assertEqual(0, len(self.file_display.file_table))
