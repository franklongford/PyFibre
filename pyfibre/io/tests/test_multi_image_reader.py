import numpy as np

from pyfibre.io.multi_image_reader import MultiImageReader, get_image_data
from pyfibre.tests.fixtures import (
    test_shg_image_path, test_shg_pl_trans_image_path)
from pyfibre.tests.probe_classes import ProbeFixedStackImage
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class ProbeMultiImageReader(MultiImageReader):

    _multi_image_class = ProbeFixedStackImage

    def create_image_stack(self, filenames):
        return [np.ones((100, 100))] * len(filenames)


class TestMultiImageReader(PyFibreTestCase):

    def setUp(self):
        self.reader = ProbeMultiImageReader()
        self.filenames = [test_shg_image_path,
                          test_shg_pl_trans_image_path]

    def test_get_image_data(self):
        test_image = np.zeros((100, 100))
        self.assertEqual((None, 1, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((100, 100, 3))
        self.assertEqual((2, 1, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((3, 100, 100))
        self.assertEqual((None, 3, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((4, 100, 100, 3))
        self.assertEqual((None, 4, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((4, 3, 100, 100))
        self.assertEqual((1, 4, (100, 100)), get_image_data(test_image))

        test_image = np.zeros((2, 100, 100, 3, 4))
        with self.assertRaises(IndexError):
            get_image_data(test_image)

    def test_load_images(self):
        images = self.reader._load_images(self.filenames)
        self.assertEqual(2, len(images))
        self.assertEqual((200, 200), images[0].shape)
        self.assertEqual((3, 200, 200), images[1].shape)

    def test__format_image(self):

        image = self.reader._format_image(
            np.ones((10, 10)) * 2
        )
        self.assertArrayAlmostEqual(
            np.ones((10, 10)),
            image
        )
        self.assertEqual(np.float, image.dtype)

        image = self.reader._format_image(
            np.ones((10, 10, 3)) * 2
        )
        self.assertArrayAlmostEqual(
            np.ones((10, 10, 3)),
            image
        )

        image = self.reader._format_image(
            np.ones((10, 10, 3)) * 2, 2
        )
        self.assertArrayAlmostEqual(
            np.ones((10, 10)),
            image
        )

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(
            [test_shg_image_path])

        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(1, len(multi_image))

        with self.assertRaises(ImportError):
            self.reader.load_multi_image(self.filenames)
