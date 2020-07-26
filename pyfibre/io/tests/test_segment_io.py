from unittest import TestCase, mock
from tempfile import TemporaryDirectory
import os

import numpy as np

from skimage.measure import label, regionprops

from pyfibre.io.region_io import (
    save_regions, load_regions
)


class TestSegmentIO(TestCase):

    def setUp(self):

        self.N = 20

        self.image = np.zeros((self.N, self.N))
        for i in range(2):
            self.image += 2 * np.eye(self.N, self.N, k=5-i)
            self.image += np.rot90(np.eye(self.N, self.N, k=5 - i))

        self.binary = np.where(self.image, 1, 0)
        self.label_image = label(self.binary)
        self.segments = regionprops(self.label_image,
                                    intensity_image=self.image)

    def test_save_image(self):

        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, 'test')
            save_regions(self.segments, tmp_file, self.image.shape, 'segment')
            self.assertTrue(os.path.exists(f'{tmp_file}_segment.npy'))

            test_masks = np.load(f'{tmp_file}_segment.npy')

            self.assertEqual(test_masks.dtype, int)
            self.assertEqual(test_masks.shape, (1, self.N, self.N))
            self.assertTrue(np.allclose(
                np.where(self.image > 0, 1, 0),
                test_masks
            ))

    def test_load_label_image(self):

        def mock_load(*args):
            return np.array([self.binary])

        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, 'test')
            with mock.patch('numpy.load', mock_load, create=True):
                test_segment = load_regions(tmp_file, 'segment')

            self.assertEqual(len(self.segments), len(test_segment))

            test_bbox = test_segment[0].bbox

            for i in range(4):
                self.assertEqual(
                    self.segments[0].bbox[i], test_bbox[i])

            self.assertTrue(
                np.all(self.segments[0].image == test_segment[0].image)
            )

            with mock.patch('numpy.load', mock_load, create=True):
                test_segment = load_regions(
                    tmp_file, 'segment', image=self.image)

            self.assertAlmostEqual(
                0,
                np.abs(self.segments[0].intensity_image
                       - test_segment[0].intensity_image).sum(),
                6
            )
