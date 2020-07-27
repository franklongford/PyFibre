import os
from unittest import TestCase

from ..utils import (
    extract_prefix,
    get_files_prefixes,
    filter_input_files,
    get_image_type)


class TestReader(TestCase):

    def setUp(self):
        self.input_files = [
            os.path.join('directory', 'prefix1-pl-shg-test.png'),
            os.path.join('directory', 'prefix2-pl-shg-test.tif'),
            os.path.join('directory', 'prefix-pl-test.tif'),
            os.path.join('directory', 'prefix-shg-test.tif'),
            os.path.join('directory', 'prefix-display-test.tif'),
            os.path.join('directory', 'prefix-shg-virada.tif'),
            os.path.join('directory', 'prefix-shg-asterisco.tif')]
        self.prefix = os.path.join('directory', 'prefix')

    def test_get_image_type(self):
        self.assertEqual(
            'SHG-PL-Trans', get_image_type(self.input_files[0]))
        self.assertEqual(
            'PL-Trans', get_image_type(self.input_files[2]))
        self.assertEqual(
            'SHG', get_image_type(self.input_files[3]))

        # Test failure
        self.assertEqual(
            'Unknown', get_image_type('some-psh-test.tif'))

    def test_extract_prefix(self):
        self.assertEqual(
            f'{self.prefix}1',
            extract_prefix(self.input_files[0], '-pl-shg'))
        self.assertEqual(
            self.prefix,
            extract_prefix(self.input_files[2], '-pl'))
        self.assertEqual(
            self.prefix,
            extract_prefix(self.input_files[3], '-shg'))

    def test_get_files_prefixes(self):
        input_files = [
            self.input_files[0],
            self.input_files[2],
            self.input_files[3]
        ]

        files, prefixes = get_files_prefixes(input_files, 'SHG')
        self.assertListEqual(
            [input_files[2]], files)
        self.assertListEqual(
            [self.prefix], prefixes)

        files, prefixes = get_files_prefixes(input_files, 'PL-Trans')
        self.assertListEqual(
            [input_files[1]], files)
        self.assertListEqual(
            [self.prefix], prefixes)

        files, prefixes = get_files_prefixes(input_files, 'SHG-PL-Trans')
        self.assertListEqual(
            [input_files[0]], files)
        self.assertListEqual(
            [f'{self.prefix}1'], prefixes)

    def test_filter_input_files(self):
        input_files = self.input_files[0:2] + self.input_files[5:]

        filtered_files = filter_input_files(input_files)

        self.assertListEqual(
            [os.path.join('directory', 'prefix2-pl-shg-test.tif'),
             os.path.join('directory', 'prefix-shg-asterisco.tif')],
            filtered_files)
