import numpy as np
from unittest import TestCase

from pyfibre.fibers.convertors import (
    networks_to_binary,
)
from pyfibre.testing.example_objects import (
    generate_image,
    generate_probe_graph,
    generate_regions,
)


class TestConvertors(TestCase):
    def setUp(self):
        (self.image, self.labels, self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()
        self.regions = generate_regions()

    def test_networks_to_binary(self):
        binary = networks_to_binary(
            [self.network],
            self.image.shape,
            iterations=1,
            sigma=None,
            area_threshold=50,
        )
        self.assertEqual((10, 10), binary.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [1, 2],
                    [1, 3],
                    [2, 1],
                    [2, 2],
                    [2, 3],
                    [2, 4],
                    [3, 2],
                    [3, 3],
                ]
            ),
            np.argwhere(binary),
        )
