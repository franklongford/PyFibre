from unittest import TestCase, mock
from functools import partial

import numpy as np

from pyfibre.io.multi_image_reader import MultiImageReader

LOAD_IMAGE_PATH = 'pyfibre.io.multi_image_reader.MultiImageReader.load_images'


class ProbeMultiImageReader(MultiImageReader):

    def image_preprocessing(self, images):
        return [np.ones((100, 100, 3))] * len(self.filenames)


class TestMultiImageReader(TestCase):

    def setUp(self):
        self.reader = ProbeMultiImageReader()
        self.reader.filenames = [
            'some/path/to/a/file-pl-shg.tif',
            'some/path/to/another/file-pl.tif',
            'some/path/to/another/file-shg.tif']

    def test_load_multi_image(self):

        with mock.patch(
                LOAD_IMAGE_PATH, mock.mock_open()):
            self.reader.load_multi_image()
