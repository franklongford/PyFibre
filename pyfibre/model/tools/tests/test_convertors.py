from unittest import TestCase

import numpy as np

from pyfibre.model.tools.convertors import (
    binary_to_stack, segments_to_binary, binary_to_segments,
    networks_to_binary, stack_to_binary, stack_to_segments,
    segments_to_stack)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)


class TestConvertors(TestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

    def test_binary_to_stack(self):

        binary_stack = binary_to_stack(self.binary)
        self.assertEqual((2, 10, 10), binary_stack.shape)
        self.assertEqual(9, binary_stack[0].sum())
        self.assertEqual(3, binary_stack[1].sum())

    def test_stack_to_binary(self):

        binary = stack_to_binary(self.stack)
        self.assertEqual((10, 10), binary.shape)
        self.assertTrue(
            np.allclose(self.binary, binary)
        )

    def test_binary_to_segments(self):
        segments = binary_to_segments(self.binary, self.image)
        self.assertEqual(2, len(segments))
        self.assertEqual(9, segments[0].filled_area)
        self.assertEqual(3, segments[1].filled_area)

        segments = binary_to_segments(
            self.binary, self.image, min_size=4)
        self.assertEqual(1, len(segments))
        self.assertEqual(9, segments[0].filled_area)

        segments = binary_to_segments(
            self.binary, self.image, min_frac=3.6)
        self.assertEqual(1, len(segments))
        self.assertEqual(3, segments[0].filled_area)

        segments = binary_to_segments(self.stack, self.image)
        self.assertEqual(2, len(segments))
        self.assertEqual(9, segments[0].filled_area)
        self.assertEqual(3, segments[1].filled_area)

    def test_segments_to_binary(self):
        segments = binary_to_segments(self.binary, self.image)
        binary = segments_to_binary(segments, (10, 10))

        self.assertEqual((10, 10), binary.shape)
        self.assertTrue((self.binary == binary).all())

    def test_networks_to_binary(self):

        binary = networks_to_binary([self.network], self.image.shape,
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

    def test_networks_to_segments(self):
        pass

    def test_segments_to_stack(self):
        segments = stack_to_segments(self.stack)
        stack = segments_to_stack(segments, (10, 10))
        self.assertTrue(
            np.allclose(self.stack, stack)
        )

    def test_stack_to_segments(self):
        segments = stack_to_segments(self.stack)
        self.assertEqual(2, len(segments))
        self.assertEqual(9, segments[0].filled_area)
        self.assertEqual(3, segments[1].filled_area)
