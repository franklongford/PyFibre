import numpy as np

from skimage.external.tifffile import TiffFile

from pyfibre.io.base_multi_image_reader import (
    BaseMultiImageReader)
from pyfibre.io.shg_pl_reader import (
    get_fluoview_param,
    get_imagej_param,
    get_tiff_param
)
from pyfibre.tests.fixtures import (
    test_shg_image_path, test_shg_pl_trans_image_path)
from pyfibre.tests.probe_classes.multi_images import ProbeFixedStackImage
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class ProbeMultiImageReader(BaseMultiImageReader):

    _multi_image_class = ProbeFixedStackImage

    def can_load(self, filename):
        return True

    def load_image(self, filename):
        return np.ones((100, 100))

    def create_image_stack(self, filenames):
        return [self.load_image(filename) for filename in filenames]


class TestMultiImageReader(PyFibreTestCase):

    def setUp(self):
        self.reader = ProbeMultiImageReader()
        self.filenames = [test_shg_image_path,
                          test_shg_pl_trans_image_path]

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(
            [test_shg_image_path], 'test-pyfibre')

        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(1, len(multi_image))
        self.assertEqual('test-pyfibre', multi_image.name)
        self.assertEqual('', multi_image.path)

        with self.assertRaises(ImportError):
            self.reader.load_multi_image(self.filenames, None)

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
