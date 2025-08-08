import importlib.resources
from tifffile import TiffFile

from pyfibre.testing.pyfibre_test_case import PyFibreTestCase
from pyfibre.multi_image.tiff_utilities import (
    get_accumulation_number,
    get_fluoview_param,
    get_imagej_param,
    get_tiff_param,
)


class TestTiffUtilities(PyFibreTestCase):
    def setUp(self):
        testing_dir = self.enterContext(importlib.resources.path("pyfibre.testing"))
        self.test_shg_image_path = str(
            testing_dir / "fixtures" / "test-pyfibre-shg-Stack.tif"
        )
        self.test_shg_pl_trans_image_path = str(
            testing_dir / "fixtures" / "test-pyfibre-pl-shg-Stack.tif"
        )

    def test_fluoview_param(self):
        description = "Gamma=1\nGamma=2\n"
        xy_dim = (20, 20)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 20, 20)
        )
        self.assertIsNone(minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 3, 20, 20)
        )
        self.assertEqual(1, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_fluoview_param(
            description, xy_dim, (2, 20, 20, 2)
        )
        self.assertEqual(3, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

    def test_get_imagej_param(self):
        description = "images=1\nslices=3\n"
        xy_dim = (20, 20)

        minor_axis, n_modes, xy_dim = get_imagej_param(description, xy_dim, (3, 20, 20))
        self.assertEqual(0, minor_axis)
        self.assertEqual(1, n_modes)
        self.assertEqual((20, 20), xy_dim)

        minor_axis, n_modes, xy_dim = get_imagej_param(
            description, xy_dim, (2, 3, 20, 20)
        )
        self.assertEqual(1, minor_axis)
        self.assertEqual(2, n_modes)
        self.assertEqual((20, 20), xy_dim)

    def test_get_tiff_param(self):
        with TiffFile(self.test_shg_image_path) as tiff_file:
            minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)
            self.assertEqual(2, minor_axis)
            self.assertEqual(1, n_modes)
            self.assertEqual((200, 200), xy_dim)

        with TiffFile(self.test_shg_pl_trans_image_path) as tiff_file:
            minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)
            self.assertEqual(1, minor_axis)
            self.assertEqual(3, n_modes)
            self.assertEqual((200, 200), xy_dim)

    def test_get_accumulation_number(self):
        self.assertEqual(2, get_accumulation_number("some-file-acc2.tif"))
        self.assertEqual(12, get_accumulation_number("some-file-acc12.tif"))
        self.assertEqual(12, get_accumulation_number("some-path/some-file-acc12.tif"))
        self.assertEqual(1, get_accumulation_number("some-file.tif"))
