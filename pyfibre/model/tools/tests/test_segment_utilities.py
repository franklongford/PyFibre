from unittest import TestCase

import numpy as np
from skimage.measure import regionprops

from pyfibre.model.tools.segment_utilities import (
    draw_network, segment_check, segment_swap,
    mean_binary
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)


class TestSegmentUtilities(TestCase):

    def setUp(self):
        self.image, self.labels, self.binary = generate_image()
        self.network = generate_probe_graph()

    def test_segment_check(self):
        segment = regionprops(
            self.labels, intensity_image=self.image)[0]

        self.assertTrue(segment_check(segment))
        self.assertFalse(segment_check(segment, min_size=10))
        self.assertFalse(segment_check(segment, min_frac=3.6))
        self.assertFalse(segment_check(segment, edges=True))

    def test_segment_swap(self):

        mask_1 = self.binary.astype(bool)
        mask_2 = np.zeros(self.binary.shape).astype(bool)

        self.assertEqual(12, np.sum(mask_1.astype(int)))
        self.assertEqual(0, np.sum(mask_2.astype(int)))

        segment_swap([mask_1, mask_2], [self.image, self.image], [4, 0], [0, 0])

        self.assertEqual(9, np.sum(mask_1.astype(int)))
        self.assertEqual(3, np.sum(mask_2.astype(int)))

    def test_draw_network(self):

        label_image = np.zeros(self.image.shape, dtype=int)
        draw_network(self.network, label_image, 1)

        self.assertTrue(
            np.allclose(
                np.array([[0, 0], [1, 1], [2, 2],
                          [2, 3]]),
                np.argwhere(label_image)
            )
        )

    def test_mean_binary(self):
        binaries = np.array([
            self.binary, np.identity(self.binary.shape[0])])

        binary = mean_binary(binaries, self.image)

        self.assertEqual(self.binary.shape, binary.shape)
        self.assertEqual(37, np.sum(binary))
