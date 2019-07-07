from unittest import mock, TestCase
import numpy as np
import os

from skimage.measure import label, regionprops

from pyfibre.io.segment_io import (
    save_segment, load_segment
)

SAVE_SEGMENT_PATH = 'pyfibre.io.segment_io.save_segment'


class TestSegmentIO(TestCase):

    def setUp(self):

        self.N = 10

        self.image = np.zeros((self.N, self.N))
        for i in range(2):
            self.image += 2 * np.eye(self.N, self.N, k=5-i)

        self.label_image = label(self.image)
        self.segments = regionprops(self.label_image,
                                    intensity_image=self.image)

    def test_save_image(self):

        try:
            save_segment(self.segments, 'test')
            self.assertTrue(os.path.exists('test_.npy'))

            X = np.load('test_.npy', mmap_mode='r')

            self.assertEqual(X.dtype, int)
            self.assertEqual(X.shape, (1, self.N, self.N))
            self.assertAlmostEqual(
                0,
                np.abs(X[0] - self.label_image).sum(),
                6)

        finally:
            if os.path.exists('test_.npy'):
                os.remove('test_.npy')

    def test_load_label_image(self):

        try:
            save_segment(self.segments, 'test')
            self.assertTrue(os.path.exists('test_.npy'))

            test_segment = load_segment('test')

            self.assertEqual(len(self.segments), len(test_segment))

            test_bbox = test_segment[0].bbox

            for i in range(4):
                self.assertEqual(
                    self.segments[0].bbox[i], test_bbox[i])

            self.assertTrue(
                np.all(self.segments[0].image == test_segment[0].image)
            )

            test_segment = load_segment('test', image=self.image)

            self.assertAlmostEqual(
                0,
                np.abs(self.segments[0].intensity_image
                       - test_segment[0].intensity_image).sum(),
                6
            )
        except Exception as e:
            raise e

        finally:
            if os.path.exists('test_.npy'):
                os.remove('test_.npy')
