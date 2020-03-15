import numpy as np
from skimage.measure import regionprops

from pyfibre.model.tools.utilities import (
    draw_network, region_check, region_swap,
    mean_binary
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestSegmentUtilities(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

    def test_segment_check(self):
        segment = regionprops(
            self.labels, intensity_image=self.image)[0]

        self.assertTrue(region_check(segment))
        self.assertFalse(region_check(segment, min_size=10))
        self.assertFalse(region_check(segment, min_frac=3.6))
        self.assertFalse(region_check(segment, edges=True))

    def test_segment_swap(self):

        mask_1 = self.binary.astype(bool)
        mask_2 = np.zeros(self.binary.shape).astype(bool)

        self.assertEqual(12, np.sum(mask_1.astype(int)))
        self.assertEqual(0, np.sum(mask_2.astype(int)))

        region_swap(
            [mask_1, mask_2], [self.image, self.image],
            [4, 0], [0, 0])

        self.assertEqual(9, np.sum(mask_1.astype(int)))
        self.assertEqual(3, np.sum(mask_2.astype(int)))

    def test_draw_network(self):

        label_image = np.zeros(self.image.shape, dtype=int)
        draw_network(self.network, label_image, 1)

        self.assertArrayAlmostEqual(
            np.array([[0, 0], [1, 1], [2, 2],
                      [2, 3]]),
            np.argwhere(label_image)
        )

    def test_mean_binary(self):
        binaries = np.array([
            self.binary, np.identity(self.binary.shape[0])])

        binary = mean_binary(binaries, self.image)

        self.assertEqual(self.binary.shape, binary.shape)
        self.assertEqual(37, np.sum(binary))
