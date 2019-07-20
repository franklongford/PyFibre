from unittest import mock, TestCase
import os
from pyfibre.io.utils import parse_files, parse_file_path

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


class TestUtils(TestCase):

    def setUp(self):
        self.file_name = (
            pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif'
        )
        self.directory = os.path.dirname(self.file_name)
        self.key = 'shg'

    def test_parse_file_path(self):

        file_name, directory = parse_file_path(
            '/a/path/to/some/file-pl-shg.tif')

        self.assertIsNone(file_name)
        self.assertIsNone(directory)

        file_name, directory = parse_file_path(self.file_name)
        self.assertEqual(self.file_name, file_name)
        self.assertIsNone(directory)

        file_name, directory = parse_file_path(self.directory)
        self.assertIsNone(file_name)
        self.assertEqual(self.directory, directory)

    def test_parse_files(self):

        input_files = parse_files(self.file_name, key=self.key)
        self.assertEqual(input_files[0], self.file_name)

        input_files = parse_files(directory=self.directory,
                                  key=self.key)
        self.assertEqual(input_files[0], self.file_name)
