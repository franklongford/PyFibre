import numpy as np

from skimage.external.tifffile import TiffFile

from pyfibre.io.base_multi_image_reader import (
    BaseMultiImageReader)
from pyfibre.io.shg_pl_reader import get_tiff_param
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
