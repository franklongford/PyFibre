from pyfibre.tests.fixtures import (
    test_shg_image_path, test_shg_pl_trans_image_path)
from pyfibre.tests.probe_classes.readers import ProbeMultiImageReader
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


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
