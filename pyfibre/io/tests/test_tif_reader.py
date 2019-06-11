from unittest import TestCase, mock

import numpy as np

from pyfibre.io.tif_reader import TIFReader


class TestTIFReader(TestCase):

    def setUp(self):
        self.reader = TIFReader()

    def test_get_image_type(self):

        self.assertEqual(self.reader._get_image_type('some-pl-shg-test.tif'), 'PL-SHG')
        self.assertEqual(self.reader._get_image_type('some-pl-test.tif'), 'PL')
        self.assertEqual(self.reader._get_image_type('some-shg-test.tif'), 'SHG')

    def test_get_image_type_failure(self):
        with self.assertRaises(RuntimeError):
            self.reader._get_image_type('some-psh-test.tif')

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
        self.assertFalse(self.reader._check_shape((100, 100, 3), 'PL-SHG'))
        self.assertFalse(self.reader._check_shape((2, 100, 100), 'PL-SHG'))

    def test_import_image(self):

        with mock.patch(
                'pyfibre.io.tif_reader.TIFReader._load_image') as mock_load:

            mock_load.return_value = np.ones((3, 100, 100, 10))
            image_stack = self.reader.import_image('some-pl-shg-test.tif')
            self.assertEqual(len(image_stack), 3)
            self.assertEqual(image_stack[0].shape, (100, 100))

            mock_load.return_value = np.ones((2, 100, 100, 10))
            image_stack = self.reader.import_image('some-pl-test.tif')
            self.assertEqual(len(image_stack), 2)
            self.assertEqual(image_stack[0].shape, (100, 100))

            mock_load.return_value = np.ones((2, 100, 100, 10))
            image_stack = self.reader.import_image('some-shg-test.tif')
            self.assertEqual(image_stack.shape, (100, 100))
