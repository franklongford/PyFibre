import numpy as np

from skimage.external.tifffile import TiffFile

from pyfibre.io.multi_image_reader import (
    MultiImageReader, get_tiff_param)
from pyfibre.tests.fixtures import (
    test_shg_image_path, test_shg_pl_trans_image_path)
from pyfibre.tests.probe_classes.multi_images import ProbeFixedStackImage
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

        test_image = np.ones((3, 3, 10, 10))
        test_image[0] *= 2
        test_image[2] *= 3
        image = self.reader._format_image(
            test_image, 1
        )
        self.assertArrayAlmostEqual(
            np.ones((3, 10, 10)),
            image
        )

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(
            [test_shg_image_path])

        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(1, len(multi_image))

        with self.assertRaises(ImportError):
            self.reader.load_multi_image(self.filenames)

    def test_get_tiff_param(self):

        with TiffFile(test_shg_image_path) as tiff_file:
            minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)
            self.assertEqual(2, minor_axis)
            self.assertEqual(1, n_modes)
            self.assertEqual((200, 200), xy_dim)

        with TiffFile(test_shg_pl_trans_image_path) as tiff_file:
            minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)
            self.assertEqual(1, minor_axis)
            self.assertEqual(3, n_modes)
            self.assertEqual((200, 200), xy_dim)
