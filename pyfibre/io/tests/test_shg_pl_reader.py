from unittest import TestCase, mock

import numpy as np

from pyfibre.io.shg_pl_reader import (
    get_image_type, extract_prefix, get_files_prefixes,
    filter_input_files, populate_image_dictionary,
    collate_image_dictionary, SHGPLReader)


LOAD_IMAGE_PATH = 'pyfibre.io.shg_pl_reader.load_image'


class TestImageReader(TestCase):

    def test_get_image_type(self):
        self.assertEqual('PL-SHG', get_image_type('some-pl-shg-test.tif'))
        self.assertEqual('PL', get_image_type('some-pl-test.tif'))
        self.assertEqual('SHG', get_image_type('some-shg-test.tif'))

        # Test failure
        self.assertEqual('Unknown', get_image_type('some-psh-test.tif'))

    def test_extract_prefix(self):
        self.assertEqual('/directory/prefix',
                         extract_prefix('/directory/prefix-pl-shg-test.tif', '-pl-shg'))
        self.assertEqual('/directory/prefix',
                         extract_prefix('/directory/prefix-pl-test.tif', '-pl'))
        self.assertEqual('/directory/prefix',
                         extract_prefix('/directory/prefix-shg-test.tif', '-shg'))

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
            {'/directory/prefix': {
                'PL-SHG': '/directory/prefix-pl-shg-test.tif',
                'PL': '/directory/prefix-pl-test.tif'}},
            image_dict)
        self.assertListEqual(
            ['/directory/prefix-shg-test.tif'],
            input_files)

    def test_collate_image_dictionary(self):
        input_files = ['/directory/prefix-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif',
                       '/directory/prefix-pl-display.tif',]

        image_dict = collate_image_dictionary(input_files)

        self.assertDictEqual(
            {'/directory/prefix': {
                'PL-SHG': '/directory/prefix-pl-shg-test.tif',
                'PL': '/directory/prefix-pl-test.tif',
                'SHG': '/directory/prefix-shg-test.tif'}},
            image_dict)
        self.assertEqual(4, len(input_files))


class TestSHGPLReader(TestCase):

    def setUp(self):
        self.reader = SHGPLReader()
        self.input_files = ['some/path/to/a/file-pl-shg.tif',
                            'some/path/to/another/file-pl.tif',
                            'some/path/to/another/file-shg.tif']

    def test_check_dimension(self):

        self.assertTrue(self.reader._check_dimension(4, 'PL-SHG'))
        self.assertTrue(self.reader._check_dimension(4, 'PL'))
        self.assertTrue(self.reader._check_dimension(4, 'SHG'))
        self.assertTrue(self.reader._check_dimension(3, 'PL-SHG'))
        self.assertTrue(self.reader._check_dimension(3, 'PL'))
        self.assertTrue(self.reader._check_dimension(3, 'SHG'))
        self.assertTrue(self.reader._check_dimension(2, 'SHG'))

    def test_check_dimension_failure(self):

        self.assertFalse(self.reader._check_dimension(2, 'PL-SHG'))
        self.assertFalse(self.reader._check_dimension(2, 'PL'))

    def test_check_shape(self):

        self.assertTrue(self.reader._check_shape((3, 7, 100, 100), 'PL-SHG'))
        self.assertTrue(self.reader._check_shape((3, 100, 100, 7), 'PL-SHG'))

        self.assertTrue(self.reader._check_shape((3, 100, 100), 'PL-SHG'))
        self.assertTrue(self.reader._check_shape((100, 100, 3), 'PL-SHG'))
        self.assertTrue(self.reader._check_shape((3, 100, 100), 'SHG'))

        self.assertTrue(self.reader._check_shape((2, 100, 100), 'PL'))
        self.assertTrue(self.reader._check_shape((3, 100, 100), 'SHG'))

        self.assertTrue(self.reader._check_shape((2, 100, 100, 3), 'PL'))
        self.assertTrue(self.reader._check_shape((2, 100, 100, 3), 'SHG'))
        self.assertTrue(self.reader._check_shape((100, 100, 3), 'SHG'))

    def test_check_shape_failure(self):

        self.assertFalse(self.reader._check_shape((7, 3, 100, 100), 'PL-SHG'))
        self.assertFalse(self.reader._check_shape((3, 100, 100, 2), 'PL'))
        self.assertFalse(self.reader._check_shape((3, 100, 100, 7), 'SHG'))
        self.assertFalse(self.reader._check_shape((2, 100, 100), 'PL-SHG'))

    def test_import_image(self):

        with mock.patch(LOAD_IMAGE_PATH) as mock_load:

            mock_load.return_value = np.ones((3, 100, 100, 10))
            image_stack = self.reader.import_image(self.input_files[0], 'PL-SHG')
            self.assertEqual(len(image_stack), 3)
            self.assertEqual(image_stack[0].shape, (100, 100))

            mock_load.return_value = np.ones((2, 100, 100, 10))
            image_stack = self.reader.import_image(self.input_files[1], 'PL')
            self.assertEqual(len(image_stack), 2)
            self.assertEqual(image_stack[0].shape, (100, 100))

            mock_load.return_value = np.ones((2, 100, 100, 10))
            image_stack = self.reader.import_image(self.input_files[2], 'SHG')
            self.assertEqual(image_stack.shape, (100, 100))

    def test_get_image_lists(self):

        self.reader.get_image_lists(self.input_files)

        self.assertEqual(len(self.reader.files.keys()), 2)
        self.assertEqual(
            self.reader.files['some/path/to/a/file']['PL-SHG'],
            'some/path/to/a/file-pl-shg.tif')
        self.assertEqual(
            self.reader.files['some/path/to/another/file']['PL'],
            'some/path/to/another/file-pl.tif')
        self.assertEqual(
            self.reader.files['some/path/to/another/file']['SHG'],
            'some/path/to/another/file-shg.tif')

    def test_update_multi_images(self):

        self.reader.get_image_lists(self.input_files[:1])

        with mock.patch(LOAD_IMAGE_PATH) as mock_load:
            mock_load.return_value = np.ones((3, 100, 100))
            self.reader.load_multi_images()

        multi_image = self.reader.files['some/path/to/a/file']['image']
        self.assertEqual(self.reader.ow_network, multi_image.ow_network)
        self.assertEqual(self.reader.ow_segment, multi_image.ow_segment)
        self.assertEqual(self.reader.ow_metric, multi_image.ow_metric)
        self.assertEqual(self.reader.ow_figure, multi_image.ow_figure)

        self.reader.ow_network = True
        self.assertNotEqual(self.reader.ow_network, multi_image.ow_network)

        self.reader.update_multi_images()
        self.assertEqual(self.reader.ow_network, multi_image.ow_network)

    def test_load_multi_images(self):

        self.reader.get_image_lists(self.input_files[:1])

        with mock.patch(LOAD_IMAGE_PATH) as mock_load:
            mock_load.return_value = np.ones((3, 100, 100))
            self.reader.load_multi_images()
            mock_load.assert_called()

            multi_image = self.reader.files['some/path/to/a/file']['image']

            self.assertIsNotNone(multi_image.image_shg)
            self.assertIsNotNone(multi_image.image_pl)
            self.assertIsNotNone(multi_image.image_tran)
            self.assertEqual(
                (100, 100),
                multi_image.shape,

            )

    def test_view_shg_pl_reader(self):
        self.reader.configure_traits()