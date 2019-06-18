from unittest import TestCase
import numpy as np

from pyfibre.tools.segmentation import (
    create_binary_image, get_segments
)


class TestSegmentation(TestCase):

    def setUp(self):

        self.image = np.zeros((5, 5))
        self.image[0:3, 2] += 2
        self.image[1, 2:4] += 5

        self.binary = np.zeros((5, 5))
        self.binary[0:3, 2] = 1
        self.binary[1, 2:4] = 1

    def test_segment_image(self):
        segments = get_segments(self.image, self.binary)

        self.assertEqual(len(segments), 1)

    def test_create_binary_image(self):
        segments = get_segments(self.image, self.binary)
        binary = create_binary_image(segments, (5, 5))

        self.assertTrue((self.binary == binary).all())
