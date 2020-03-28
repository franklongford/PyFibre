from unittest import TestCase

import numpy as np

from pyfibre.io.shg_pl_reader import (
    get_image_type, get_image_data, extract_prefix,
    get_files_prefixes, filter_input_files,
    populate_image_dictionary,
    collate_image_dictionary, SHGPLReader,
    SHGPLTransReader
)


class TestImageReader(TestCase):

    def test_get_image_type(self):
        self.assertEqual(
            'PL-SHG', get_image_type('some-pl-shg-test.tif'))
        self.assertEqual(
            'PL', get_image_type('some-pl-test.tif'))
        self.assertEqual(
            'SHG', get_image_type('some-shg-test.tif'))

        # Test failure
        self.assertEqual(
            'Unknown', get_image_type('some-psh-test.tif'))

    def test_get_image_data(self):

        test_image = np.zeros((100, 100))
        self.assertEqual((None, 1, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((100, 100, 3))
        self.assertEqual((2, 1, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((3, 100, 100))
        self.assertEqual((None, 3, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((4, 100, 100, 3))
        self.assertEqual((3, 4, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((2, 100, 100, 3))
        self.assertEqual((3, 2, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((2, 100, 100, 3, 4))
        with self.assertRaises(IndexError):
            get_image_data(test_image)

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
                       '/directory/prefix2-pl-shg-test.tif',
                       '/directory/prefix-shg-test.tif']

        files, prefixes = get_files_prefixes(input_files, '-pl-shg')
        self.assertListEqual(
            ['/directory/prefix1-pl-shg-test.tif',
             '/directory/prefix2-pl-shg-test.tif'], files)
        self.assertListEqual(
            ['/directory/prefix1', '/directory/prefix2'], prefixes)

    def test_filter_input_files(self):
        input_files = ['/directory/prefix1-pl-shg-test.png',
                       '/directory/prefix2-pl-shg-test.tif',
                       '/directory/prefix-display-test.tif',
                       '/directory/prefix-shg-virada.tif',
                       '/directory/prefix-shg-asterisco.tif']

        filtered_files = filter_input_files(input_files)

        self.assertListEqual(
            ['/directory/prefix2-pl-shg-test.tif'], filtered_files)

    def test_populate_image_dictionary(self):
        input_files = ['/directory/prefix-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif']
        image_dict = {}

        populate_image_dictionary(input_files, image_dict, 'PL-SHG')

        self.assertDictEqual(
            {'/directory/prefix': {
                'PL-SHG': '/directory/prefix-pl-shg-test.tif'}},
            image_dict)
        self.assertListEqual(
            ['/directory/prefix-pl-test.tif',
             '/directory/prefix-shg-test.tif'],
            input_files)

        populate_image_dictionary(input_files, image_dict, 'PL')

        self.assertDictEqual(
            {
                '/directory/prefix':
                    {'PL-SHG': '/directory/prefix-pl-shg-test.tif',
                     'PL': '/directory/prefix-pl-test.tif'}
            },
            image_dict)
        self.assertListEqual(
            ['/directory/prefix-shg-test.tif'],
            input_files)

    def test_collate_image_dictionary(self):
        input_files = ['/directory/prefix-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif',
                       '/directory/prefix-pl-display.tif']

        image_dict = collate_image_dictionary(input_files)

        self.assertDictEqual(
            {'/directory/prefix': {
                'PL-SHG': '/directory/prefix-pl-shg-test.tif',
                'PL': '/directory/prefix-pl-test.tif',
                'SHG': '/directory/prefix-shg-test.tif'}},
            image_dict)
        self.assertEqual(4, len(input_files))


class ProbeSHGPLReader(SHGPLReader):

    def load_images(self):
        if self.load_mode == 'PL-SHG File':
            return [np.ones((2, 100, 100, 3))]
        return [np.ones((100, 100, 3))] * 2


class TestSHGPLReader(TestCase):

    def setUp(self):
        self.reader = ProbeSHGPLReader()
        self.input_files = ['some/path/to/a/file-pl-shg.tif',
                            'some/path/to/another/file-pl.tif',
                            'some/path/to/another/file-shg.tif']

    def test_filenames(self):
        self.reader.shg_pl_filename = self.input_files[0]
        self.reader.pl_filename = self.input_files[1]
        self.reader.shg_filename = self.input_files[2]

        self.assertListEqual(
            ['some/path/to/a/file-pl-shg.tif'],
            self.reader.filenames
        )

        self.reader.load_mode = 'Separate Files'

        self.assertListEqual(
            ['some/path/to/another/file-shg.tif',
             'some/path/to/another/file-pl.tif'],
            self.reader.filenames
        )

    def test_check_dimension(self):

        self.assertTrue(self.reader._check_dimension(4, 'PL-SHG'))
        self.assertTrue(self.reader._check_dimension(4, 'PL'))
        self.assertTrue(self.reader._check_dimension(4, 'SHG'))
        self.assertTrue(self.reader._check_dimension(3, 'PL-SHG'))
        self.assertTrue(self.reader._check_dimension(3, 'PL'))
        self.assertTrue(self.reader._check_dimension(3, 'SHG'))
        self.assertTrue(self.reader._check_dimension(2, 'SHG'))
        self.assertTrue(self.reader._check_dimension(2, 'PL'))

        self.assertFalse(self.reader._check_dimension(2, 'PL-SHG'))
        self.assertFalse(self.reader._check_dimension(5, 'PL'))
        self.assertFalse(self.reader._check_dimension(5, 'SHG'))

    def test_check_n_modes(self):

        self.assertTrue(self.reader._check_n_modes(3, 'PL-SHG'))
        self.assertTrue(self.reader._check_n_modes(2, 'SHG'))
        self.assertTrue(self.reader._check_n_modes(2, 'PL'))
        self.assertTrue(self.reader._check_n_modes(1, 'PL'))
        self.assertTrue(self.reader._check_n_modes(1, 'SHG'))

        self.assertFalse(self.reader._check_n_modes(5, 'PL-SHG'))
        self.assertFalse(self.reader._check_n_modes(4, 'PL'))
        self.assertFalse(self.reader._check_n_modes(4, 'SHG'))

    def test_format_image(self):

        test_image = np.zeros((2, 100, 100, 3))
        formatted_image = self.reader._format_image(
            test_image, 2, 3)
        self.assertEqual(2, len(formatted_image))
        self.assertEqual((100, 100), formatted_image[0].shape)

        test_image = np.zeros((100, 100, 3))
        formatted_image = self.reader._format_image(
            test_image, 1, 2)
        self.assertEqual(1, len(formatted_image))
        self.assertEqual((100, 100), formatted_image[0].shape)

        test_image = np.zeros((3, 100, 100))
        formatted_image = self.reader._format_image(
            test_image, 3, None)
        self.assertEqual(3, len(formatted_image))
        self.assertEqual((100, 100), formatted_image[0].shape)

        test_image = np.zeros((100, 100))
        formatted_image = self.reader._format_image(
            test_image, 1, None)
        self.assertEqual(1, len(formatted_image))
        self.assertEqual((100, 100), formatted_image[0].shape)

    def test_image_preprocessing(self):

        images = [np.zeros((2, 100, 100, 3))]

        processed_images = self.reader.image_preprocessing(images)
        self.assertEqual(2, len(processed_images))
        self.assertEqual((100, 100), processed_images[0].shape)

        self.reader.load_mode = 'Separate Files'
        images = [np.zeros((100, 100, 3)),
                  np.zeros((100, 100))]

        processed_images = self.reader.image_preprocessing(images)
        self.assertEqual(2, len(processed_images))
        self.assertEqual((100, 100), processed_images[0].shape)

    def test_load_multi_image(self):

        self.reader.shg_pl_filename = self.input_files[0]
        self.reader.pl_filename = self.input_files[1]
        self.reader.shg_filename = self.input_files[2]

        multi_image = self.reader.load_multi_image()
        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(2, len(multi_image))

        self.reader.load_mode = 'Separate Files'
        multi_image = self.reader.load_multi_image()
        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(2, len(multi_image))

    def test_assign_images(self):

        image_dictionary = {
            'PL-SHG': '/directory/prefix-pl-shg-test.tif',
            'PL': '/directory/prefix-pl-test.tif',
            'SHG': '/directory/prefix-shg-test.tif'}

        self.reader.assign_images(image_dictionary)

        self.assertEqual(
            '/directory/prefix-pl-shg-test.tif',
            self.reader.shg_pl_filename)
        self.assertEqual(
            '/directory/prefix-pl-test.tif',
            self.reader.pl_filename)
        self.assertEqual(
            '/directory/prefix-shg-test.tif',
            self.reader.shg_filename)


class ProbeSHGPLTransReader(SHGPLTransReader):

    def load_images(self):
        if self.load_mode == 'PL-SHG File':
            return [np.ones((3, 100, 100, 3))]
        return [np.ones((2, 100, 100, 3))] * 2


class TestSHGPLTransReader(TestCase):

    def setUp(self):
        self.reader = ProbeSHGPLTransReader()
        self.input_files = ['some/path/to/a/file-pl-shg.tif',
                            'some/path/to/another/file-pl.tif',
                            'some/path/to/another/file-shg.tif']

    def test_image_preprocessing(self):

        images = [np.zeros((3, 100, 100, 3))]

        processed_images = self.reader.image_preprocessing(images)
        self.assertEqual(3, len(processed_images))
        self.assertEqual((100, 100), processed_images[0].shape)

        self.reader.load_mode = 'Separate Files'
        images = [np.zeros((2, 100, 100, 3)),
                  np.zeros((2, 100, 100))]

        processed_images = self.reader.image_preprocessing(images)
        self.assertEqual(3, len(processed_images))
        self.assertEqual((100, 100), processed_images[0].shape)

    def test_load_multi_image(self):

        self.reader.shg_pl_filename = self.input_files[0]
        self.reader.pl_filename = self.input_files[1]
        self.reader.shg_filename = self.input_files[2]

        multi_image = self.reader.load_multi_image()
        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(3, len(multi_image))

        self.reader.load_mode = 'Separate Files'
        multi_image = self.reader.load_multi_image()
        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(3, len(multi_image))
