from unittest import TestCase
import os

import numpy as np

from skimage.measure import label, regionprops

from pyfibre.io.segment_io import (
    save_segments, load_segments
)

SAVE_SEGMENT_PATH = 'pyfibre.io.segment_io.save_segments'


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

        try:
            save_segments(self.segments, 'test', self.image.shape, 'segment')
            self.assertTrue(os.path.exists('test_segment.npy'))

            test_masks = np.load('test_segment.npy', mmap_mode='r')

            self.assertEqual(test_masks.dtype, int)
            self.assertEqual(test_masks.shape, (self.N, self.N))
            self.assertTrue(np.allclose(
                np.where(self.image > 0, 1, 0),
                test_masks
            ))

        finally:
            if os.path.exists('test_segment.npy'):
                os.remove('test_segment.npy')

    def test_load_label_image(self):

        try:
            save_segments(self.segments, 'test', self.image.shape, 'segment')
            self.assertTrue(os.path.exists('test_segment.npy'))

            test_segment = load_segments('test', 'segment')

            self.assertEqual(len(self.segments), len(test_segment))

            test_bbox = test_segment[0].bbox

            for i in range(4):
                self.assertEqual(
                    self.segments[0].bbox[i], test_bbox[i])

            self.assertTrue(
                np.all(self.segments[0].image == test_segment[0].image)
            )

            test_segment = load_segments('test', 'segment', image=self.image)

            self.assertAlmostEqual(
                0,
                np.abs(self.segments[0].intensity_image
                       - test_segment[0].intensity_image).sum(),
                6
            )
        except Exception as e:
            raise e

        finally:
            if os.path.exists('test_segment.npy'):
                os.remove('test_segment.npy')
