from unittest import TestCase

import networkx as nx
import numpy as np
from skimage.measure import label, regionprops

from pyfibre.model.tools.segment_utilities import (
    segments_to_binary,
    draw_network, segment_check, binary_to_segments,
    binary_to_stack, networks_to_binary
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_network
)


class TestSegmentUtilities(TestCase):

    def setUp(self):
        self.image, self.labels, self.binary = generate_image()
        self.network = generate_probe_network()

    def test_binary_to_stack(self):

        binary_stack = binary_to_stack(self.binary)
        self.assertEqual((2, 10, 10), binary_stack.shape)
        self.assertEqual(9, binary_stack[0].sum())
        self.assertEqual(3, binary_stack[1].sum())

    def test_segment_check(self):
        segment = regionprops(self.labels, intensity_image=self.image,
                              coordinates='xy')[0]

        self.assertTrue(segment_check(segment))
        self.assertFalse(segment_check(segment, min_size=10))
        self.assertFalse(segment_check(segment, min_frac=3.6))
        self.assertFalse(segment_check(segment, edges=True))

    def test_binary_to_segments(self):
        segments = binary_to_segments(self.binary, self.image)
        self.assertEqual(2, len(segments))
        self.assertEqual(9, segments[0].filled_area)
        self.assertEqual(3, segments[1].filled_area)

        segments = binary_to_segments(self.binary, self.image, min_size=4)
        self.assertEqual(1, len(segments))
        self.assertEqual(9, segments[0].filled_area)

        segments = binary_to_segments(self.binary, self.image, min_frac=3.6)
        self.assertEqual(1, len(segments))
        self.assertEqual(3, segments[0].filled_area)

        binary_stack = binary_to_stack(self.binary)
        segments = binary_to_segments(binary_stack, self.image)
        self.assertEqual(2, len(segments))
        self.assertEqual(9, segments[0].filled_area)
        self.assertEqual(3, segments[1].filled_area)

    def test_segments_to_binary(self):
        segments = binary_to_segments(self.binary, self.image)
        binary = segments_to_binary(segments, (10, 10))

        self.assertEqual((10, 10), binary.shape)
        self.assertTrue((self.binary == binary).all())

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

    def test_networks_to_binary(self):

        binary = networks_to_binary([self.network], self.image,
                                    iterations=1, sigma=None,
                                    area_threshold=50)
        self.assertEqual((10, 10), binary.shape)
        self.assertTrue(
            np.allclose(
                np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                          [1, 2], [1, 3], [2, 1], [2, 2],
                          [2, 3], [2, 4], [3, 2], [3, 3]]),
                np.argwhere(binary)
            )
        )

    def test_filter_segments(self):
        pass