import numpy as np

from pyfibre.tools.convertors import (
    binary_to_stack,
    regions_to_binary,
    binary_to_regions,
    stack_to_binary,
    stack_to_regions,
    regions_to_stack,
)
from pyfibre.testing.pyfibre_test_case import PyFibreTestCase
from pyfibre.testing.example_objects import (
    generate_image,
    generate_probe_graph,
    generate_regions,
)


class TestConvertors(PyFibreTestCase):
    def setUp(self):
        (self.image, self.labels, self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()
        self.regions = generate_regions()

    def test_binary_to_stack(self):
        binary_stack = binary_to_stack(self.binary)
        self.assertEqual((2, 10, 10), binary_stack.shape)
        self.assertEqual(9, binary_stack[0].sum())
        self.assertEqual(3, binary_stack[1].sum())

    def test_stack_to_binary(self):
        binary = stack_to_binary(self.stack)
        self.assertEqual((10, 10), binary.shape)
        self.assertTrue(np.allclose(self.binary, binary))

    def test_binary_to_regions(self):
        regions = binary_to_regions(self.binary, self.image)
        self.assertEqual(2, len(regions))
        self.assertEqual(9, regions[0].filled_area)
        self.assertEqual(3, regions[1].filled_area)

        regions = binary_to_regions(self.binary, self.image, min_size=4)
        self.assertEqual(1, len(regions))
        self.assertEqual(9, regions[0].filled_area)

        regions = binary_to_regions(self.binary, self.image, min_frac=3.6)
        self.assertEqual(1, len(regions))
        self.assertEqual(3, regions[0].filled_area)

        regions = binary_to_regions(self.stack, self.image)
        self.assertEqual(2, len(regions))
        self.assertEqual(9, regions[0].filled_area)
        self.assertEqual(3, regions[1].filled_area)

    def test_regions_to_binary(self):
        binary = regions_to_binary(self.regions, (10, 10))

        self.assertEqual((10, 10), binary.shape)
        self.assertTrue((self.binary == binary).all())

    def test_regions_to_stack(self):
        stack = regions_to_stack(self.regions, (10, 10))
        self.assertArrayAlmostEqual(self.stack, stack)

    def test_stack_to_regions(self):
        regions = stack_to_regions(self.stack)
        self.assertEqual(2, len(regions))
        self.assertEqual(9, regions[0].filled_area)
        self.assertEqual(3, regions[1].filled_area)
