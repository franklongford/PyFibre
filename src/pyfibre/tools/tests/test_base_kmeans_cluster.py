import numpy as np

from pyfibre.testing.example_objects import generate_image, generate_probe_graph
from pyfibre.testing.probe_objects import ProbeKmeansFilter
from pyfibre.testing.pyfibre_test_case import PyFibreTestCase


class TestKmeansFilter(PyFibreTestCase):
    def setUp(self):
        (self.image, self.labels, self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.image = np.stack([self.image] * 3, axis=-1)

        self.bd_filter = ProbeKmeansFilter(n_runs=2, n_clusters=2)

    def test_cluster_colours(self):
        labels, centres = self.bd_filter._kmeans_cluster_colours(self.image)

        self.assertArrayAlmostEqual(np.arange(2), np.unique(labels))
        self.assertEqual((2, 3), centres.shape)

    def test_create_scaled_image(self):
        image_scaled = self.bd_filter._scale_image(self.image)

        self.assertEqual((10, 10, 3), image_scaled.shape)
        self.assertEqual((10, 10), self.bd_filter._greyscale.shape)
        self.assertAlmostEqual(225.0, image_scaled.mean())

    def test__cluster_generator(self):
        n_runs = 0

        for mask, cost in self.bd_filter._cluster_generator(self.image):
            self.assertEqual(self.image.shape[:-1], mask.shape)
            self.assertAlmostEqual(1.0, cost)
            n_runs += 1

        self.assertEqual(2, n_runs)

    def test_BD_filter(self):
        mask_image = self.bd_filter.filter_image(self.image)

        self.assertEqual((10, 10), mask_image.shape)
        self.assertIsNone(self.bd_filter._greyscale)
