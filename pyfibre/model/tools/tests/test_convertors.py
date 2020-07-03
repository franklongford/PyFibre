import numpy as np

from pyfibre.model.tools.convertors import (
    binary_to_stack, regions_to_binary, binary_to_regions,
    networks_to_binary, stack_to_binary, stack_to_regions,
    regions_to_stack, binary_to_segments, segments_to_binary)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.tests.probe_classes.objects import ProbeSegment
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_probe_graph, generate_regions
)


class TestConvertors(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()
        self.regions = generate_regions()
        self.segments = [
            ProbeSegment(region=region) for region in self.regions
        ]

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

    def test_binary_to_regions(self):
        regions = binary_to_regions(self.binary, self.image)
        self.assertEqual(2, len(regions))
        self.assertEqual(9, regions[0].filled_area)
        self.assertEqual(3, regions[1].filled_area)

        regions = binary_to_regions(
            self.binary, self.image, min_size=4)
        self.assertEqual(1, len(regions))
        self.assertEqual(9, regions[0].filled_area)

        regions = binary_to_regions(
            self.binary, self.image, min_frac=3.6)
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

    def test_binary_to_segments(self):

        segments = binary_to_segments(
            self.binary, ProbeSegment)
        self.assertEqual(0, len(segments))

        segments = binary_to_segments(
            self.binary, ProbeSegment, min_size=4)
        self.assertEqual(1, len(segments))
        self.assertIsInstance(segments[0], ProbeSegment)

        expected = np.array([[2., 0., 0., 0.],
                             [2., 0., 0., 0.],
                             [7., 5., 5., 5.],
                             [2., 0., 0., 0.],
                             [2., 0., 0., 0.],
                             [2., 0., 0., 0.]])

        segments = binary_to_segments(
            self.binary, ProbeSegment,
            intensity_image=self.image, min_size=4)
        self.assertArrayAlmostEqual(
            expected, segments[0].region.intensity_image)

    def test_networks_to_binary(self):

        binary = networks_to_binary([self.network], self.image.shape,
                                    iterations=1, sigma=None,
                                    area_threshold=50)
        self.assertEqual((10, 10), binary.shape)
        self.assertArrayAlmostEqual(
            np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                      [1, 2], [1, 3], [2, 1], [2, 2],
                      [2, 3], [2, 4], [3, 2], [3, 3]]),
            np.argwhere(binary)
        )

    def test_networks_to_segments(self):
        pass

    def test_regions_to_stack(self):
        stack = regions_to_stack(self.regions, (10, 10))
        self.assertArrayAlmostEqual(self.stack, stack)

    def test_stack_to_regions(self):
        regions = stack_to_regions(self.stack)
        self.assertEqual(2, len(regions))
        self.assertEqual(9, regions[0].filled_area)
        self.assertEqual(3, regions[1].filled_area)

    def test_segments_to_binary(self):
        binary = segments_to_binary(self.segments, (10, 10))
        self.assertEqual((10, 10), binary.shape)
        self.assertArrayAlmostEqual(self.binary, binary)
