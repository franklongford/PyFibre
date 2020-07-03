from unittest import TestCase

from ..utils import (
    extract_prefix,
    get_files_prefixes,
    filter_input_files,
    get_image_type)


class TestReader(TestCase):

    def setUp(self):
        self.image_dictionary = {
            'SHG-PL-Trans': '/directory/prefix-pl-shg-test.tif',
            'PL-Trans': '/directory/prefix-pl-test.tif',
            'SHG': '/directory/prefix-shg-test.tif'}

    def test_get_image_type(self):
        self.assertEqual(
            'SHG-PL-Trans', get_image_type('some-pl-shg-test.tif'))
        self.assertEqual(
            'PL-Trans', get_image_type('some-pl-test.tif'))
        self.assertEqual(
            'SHG', get_image_type('some-shg-test.tif'))

        # Test failure
        self.assertEqual(
            'Unknown', get_image_type('some-psh-test.tif'))

    def test_extract_prefix(self):
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-pl-shg-test.tif', '-pl-shg'))
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-pl-test.tif', '-pl'))
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-shg-test.tif', '-shg'))

    def test_get_files_prefixes(self):
        input_files = ['/directory/prefix1-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif']

        files, prefixes = get_files_prefixes(input_files, 'SHG')
        self.assertListEqual(
            ['/directory/prefix-shg-test.tif'], files)
        self.assertListEqual(
            ['/directory/prefix'], prefixes)

        files, prefixes = get_files_prefixes(input_files, 'PL-Trans')
        self.assertListEqual(
            ['/directory/prefix-pl-test.tif'], files)
        self.assertListEqual(
            ['/directory/prefix'], prefixes)

        files, prefixes = get_files_prefixes(input_files, 'SHG-PL-Trans')
        self.assertListEqual(
            ['/directory/prefix1-pl-shg-test.tif'], files)
        self.assertListEqual(
            ['/directory/prefix1'], prefixes)

    def test_filter_input_files(self):
        input_files = ['/directory/prefix1-pl-shg-test.png',
                       '/directory/prefix2-pl-shg-test.tif',
                       '/directory/prefix-display-test.tif',
                       '/directory/prefix-shg-virada.tif',
                       '/directory/prefix-shg-asterisco.tif']

        filtered_files = filter_input_files(input_files)

        self.assertListEqual(
            ['/directory/prefix2-pl-shg-test.tif',
             '/directory/prefix-shg-asterisco.tif'], filtered_files)
