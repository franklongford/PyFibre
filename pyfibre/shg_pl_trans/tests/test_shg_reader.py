import numpy as np
from skimage.external.tifffile import TiffFile

from pyfibre.core.base_multi_image_reader import WrongFileTypeError
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.shg_pl_trans.tests.fixtures import (
    directory,
    test_shg_image_path,
    test_shg_pl_trans_image_path)

from ..shg_reader import (
    get_fluoview_param,
    get_imagej_param,
    get_tiff_param,
    SHGReader
)


class TestTiffReader(PyFibreTestCase):

    def test_fluoview_param(self):
        description = "Gamma=1\nGamma=2\n"
        xy_dim = (20, 20)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 20, 20))
        self.assertIsNone(minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 3, 20, 20))
        self.assertEqual(1, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 20, 20, 2))
        self.assertEqual(3, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

    def test_get_imagej_param(self):
        description = "images=1\nslices=3\n"
        xy_dim = (20, 20)

        minor_axis, n_modes, xy_dim = get_imagej_param(
            description, xy_dim, (3, 20, 20))
        self.assertEqual(0, minor_axis)
        self.assertEqual(1, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_imagej_param(
            description, xy_dim, (2, 3, 20, 20))
        self.assertEqual(1, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

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


class TestSHGReader(PyFibreTestCase):

    def setUp(self):
        self.reader = SHGReader()
        self.filenames = [test_shg_image_path]

    def test_can_load(self):

        self.assertTrue(self.reader.can_load(test_shg_image_path))
        self.assertFalse(self.reader.can_load('not_an_image'))

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

    def test_collate_files(self):
        image_dict = self.reader.collate_files(self.filenames)

        self.assertDictEqual(
            {f'{directory}/test-pyfibre': [test_shg_image_path]},
            image_dict
        )

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(
            self.filenames, '/some/path/test-shg')
        self.assertEqual('test-shg', multi_image.name)
        self.assertEqual('/some/path', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(1, len(multi_image))
