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

        with self.assertRaises(ImportError):
            self.reader._check_dimension(2, 'PL-SHG')
        with self.assertRaises(ImportError):
            self.reader._check_dimension(2, 'PL')

    def test_check_shape(self):

        image = np.zeros((3, 7, 100, 100))
        self.assertTrue(self.reader._check_shape(image, 0, 3, 'PL-SHG'))
        image = np.zeros((100, 100, 3))
        self.assertTrue(self.reader._check_shape(image, 0, 3, 'PL-SHG'))
        image = np.zeros((2, 100, 100))
        self.assertTrue(self.reader._check_shape(image, 'PL'))
        self.assertTrue(self.reader._check_shape(image, 'SHG'))
        image = np.zeros((100, 100))
        self.assertTrue(self.reader._check_shape(image, 'SHG'))

    def test_check_shape_failure(self):

        image = np.zeros((100, 100))
        with self.assertRaises(ImportError):
            self.reader._check_dimension(image, 'PL-SHG')
        with self.assertRaises(ImportError):
            self.reader._check_dimension(image, 'PL')
