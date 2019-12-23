from unittest import TestCase

import numpy as np

from pyfibre.io.multi_image_reader import MultiImageReader


class ProbeMultiImageReader(MultiImageReader):

    def image_preprocessing(self, images):
        return images

    def load_images(self):
        return [np.ones((100, 100))] * len(self.filenames)


class TestMultiImageReader(TestCase):

    def setUp(self):
        self.reader = ProbeMultiImageReader()
        self.reader.filenames = [
            'some/path/to/a/file-pl-shg.tif',
            'some/path/to/another/file-pl.tif',
            'some/path/to/another/file-shg.tif']

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image()

        self.assertEqual((100, 100), multi_image.shape)
        self.assertEqual(3, len(multi_image))
