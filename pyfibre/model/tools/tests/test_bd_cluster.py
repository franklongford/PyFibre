import numpy as np
from skimage.io import imread

from pyfibre.model.tools.bd_cluster import (
    nonzero_mean, spherical_coords, cluster_colours,
    cluster_mask, BDFilter
)
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_probe_graph
)
from pyfibre.tests.fixtures import test_shg_pl_trans_image_path
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestBDCluster(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.image_stack = imread(
            test_shg_pl_trans_image_path).mean(axis=-1)

        self.image = np.zeros(self.image_stack[0].shape + (3,))
        for index, image in enumerate(self.image_stack):
            self.image[..., index] = image / image.max()

        self.bd_filter = BDFilter()

    def test_nonzero_mean(self):

        array = np.zeros(10)
        array[3] = 2
        array[7] = 4

        self.assertEqual(3, nonzero_mean(array))

    def test_spherical_coords(self):
        array = np.linspace(0, 1, 10).repeat(3).reshape(10, 3)

        x, y, z = spherical_coords(array)

        self.assertArrayAlmostEqual(
            x,
            np.array([0.,  0.11134101, 0.22409309,
                      0.33983691, 0.46055399, 0.58903097,
                      0.72972766, 0.89112251, 1.09491408,
                      1.57079633])
        )
        self.assertArrayAlmostEqual(x, y)
        self.assertArrayAlmostEqual(
            z,
            np.array([1.57079633, 1.45945531, 1.34670323,
                      1.23095942, 1.11024234, 0.98176536,
                      0.84106867, 0.67967382, 0.47588225,
                      0.]
            )
        )

    def test_cluster_mask(self):

        centres = np.array([[0.1, 0.3, 0.9],
                            [0.8, 0.3, 0.9],
                            [0.1, 0.5, 0.9]])
        intensities = np.arange(3)
        param = (0.25, 0.5, 1.0, 2.0)

        clusters, cost = cluster_mask(
            centres, intensities, param)

        self.assertArrayAlmostEqual(
            np.zeros(1), clusters)
        self.assertAlmostEqual(0.762825597, cost)

    def test_create_scaled_image(self):
        image_scaled = self.bd_filter._scale_image(self.image)

        self.assertEqual(image_scaled.shape, self.image.shape)
        self.assertAlmostEqual(130.66309166, image_scaled.mean())

    def test_cluster_colours(self):
        labels, centres = cluster_colours(self.image)

        self.assertArrayAlmostEqual(
            np.arange(8), np.unique(labels)
        )
        self.assertEqual((8, 3), centres.shape)

    def test_BD_filter(self):
        mask_image = self.bd_filter.filter_image(self.image)

        self.assertEqual((200, 200), mask_image.shape)
