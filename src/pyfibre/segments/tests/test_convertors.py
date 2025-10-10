import numpy as np
from unittest import TestCase

from pyfibre.segments.segment import Segment
from pyfibre.segments.convertors import segments_to_binary, binary_to_segments
from pyfibre.testing.example_objects import generate_regions, generate_image


class TestConvertors(TestCase):
    def setUp(self):
        self.image, _, self.binary, _ = generate_image()
        self.regions = generate_regions()

    def test_binary_to_segments(self):
        segments = binary_to_segments(self.binary)
        self.assertEqual(0, len(segments))

    def test_binary_to_segments_with_min_size(self):
        segments = binary_to_segments(self.binary, min_size=4)
        self.assertEqual(1, len(segments))

        expected = np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [7.0, 5.0, 5.0, 5.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        )

        segments = binary_to_segments(
            self.binary, intensity_image=self.image, min_size=4
        )
        np.testing.assert_almost_equal(expected, segments[0].region.intensity_image)

    def test_segments_to_binary(self):
        segments = [Segment(region=region) for region in self.regions]
        binary = segments_to_binary(segments, (10, 10))
        np.testing.assert_almost_equal(binary, self.binary)
