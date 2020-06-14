from pyfibre.core.base_multi_image_reader import WrongFileTypeError
from pyfibre.tests.fixtures import test_image_path
from pyfibre.tests.probe_classes.readers import ProbeMultiImageReader
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestMultiImageReader(PyFibreTestCase):

    def setUp(self):
        self.reader = ProbeMultiImageReader()

    def test_create_image_stack(self):
        image_stack = self.reader.create_image_stack(
            [test_image_path])
        self.assertEqual(1, len(image_stack))
        self.assertEqual((100, 100), image_stack[0].shape)

        with self.assertRaises(WrongFileTypeError):
            self.reader.create_image_stack(['WRONG'])

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(
            [test_image_path], 'test-pyfibre')

        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(1, len(multi_image))
        self.assertEqual('test-pyfibre', multi_image.name)
        self.assertEqual('', multi_image.path)

        with self.assertRaises(ImportError):
            self.reader.load_multi_image(
                [test_image_path] * 2, None)
