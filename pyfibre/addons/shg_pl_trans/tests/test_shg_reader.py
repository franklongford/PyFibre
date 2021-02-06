import numpy as np
from skimage.external.tifffile import TiffFile

from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.addons.shg_pl_trans.shg_pl_trans_parser import (
    SHGPLTransFileSet)
from pyfibre.addons.shg_pl_trans.shg_reader import (
    get_fluoview_param,
    get_imagej_param,
    get_tiff_param,
    SHGReader
)


from .fixtures import (
    test_shg_image_path,
    test_shg_pl_trans_image_path)


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
        self.file_set = SHGPLTransFileSet(
            prefix='/some/path/test-shg',
            registry={'SHG': test_shg_image_path}
        )

    def test_can_load(self):

        self.assertTrue(self.reader.can_load(test_shg_image_path))
        self.assertFalse(self.reader.can_load('not_an_image'))

    def test__format_image(self):

        expected_2D_array = np.ones((10, 10)) * 2
        expected_3D_array = np.ones((10, 10, 3)) * 2

        image = self.reader._format_image(
            np.ones((10, 10)) * 2
        )
        self.assertArrayAlmostEqual(
            expected_2D_array,
            image
        )
        self.assertEqual(np.float, image.dtype)

        image = self.reader._format_image(
            np.ones((10, 10, 3)) * 2
        )
        self.assertArrayAlmostEqual(
            expected_3D_array,
            image
        )

        image = self.reader._format_image(
            np.ones((10, 10, 3)) * 2, 2
        )
        self.assertArrayAlmostEqual(
            expected_2D_array,
            image
        )

        test_image = np.ones((3, 3, 10, 10))
        test_image[0] *= 2
        test_image[2] *= 3
        image = self.reader._format_image(
            test_image, 1
        )
        expected_array = np.ones((3, 10, 10))
        expected_array[0] *= 2
        expected_array[2] *= 3
        self.assertArrayAlmostEqual(
            expected_array,
            image
        )

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(self.file_set)
        self.assertEqual('test-shg', multi_image.name)
        self.assertEqual('/some/path', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(1, len(multi_image))
