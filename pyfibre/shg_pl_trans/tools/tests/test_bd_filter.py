import numpy as np
from skimage.io import imread

from pyfibre.shg_pl_trans.tools.bd_filter import (
    nonzero_mean, spherical_coords,
    binary_classifier_spherical,
    distance_sum,
    SHGPLTransBDFilter
)
from pyfibre.tests.fixtures import test_shg_pl_trans_image_path
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestBDCluster(PyFibreTestCase):

    def setUp(self):
        self.image_stack = imread(
            test_shg_pl_trans_image_path).mean(axis=-1)

        self.image = np.zeros(self.image_stack[0].shape + (3,))
        for index, image in enumerate(self.image_stack):
            self.image[..., index] = image / image.max()

        self.bd_filter = SHGPLTransBDFilter()

    def test_nonzero_mean(self):

        array = np.zeros(10)
        array[3] = 2
        array[7] = 4

        self.assertEqual(3, nonzero_mean(array))

    def test_distance_sum(self):

        vector = np.ones((4, 4))
        vector[0, 1] *= 3

        cost = distance_sum(vector, (0, 1, 2, 3))

        self.assertEqual(18.0, cost)

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
                      0.])
        )

    def test_binary_classifier_spherical(self):

        centres = np.array([[0.1, 0.3, 0.9],
                            [0.8, 0.3, 0.9],
                            [0.1, 0.5, 0.9]])
        intensities = np.arange(3)
        param = (0.25, 0.5, 1.0, 2.0)

        mask, cost = binary_classifier_spherical(
            centres, intensities, param)

        self.assertArrayAlmostEqual(
            np.array([1, 0, 0]), mask)
        self.assertAlmostEqual(0.855886887, cost)

    def test_cellular_classifier(self):

        self.bd_filter._greyscale = self.image[0]

        mask, cost = self.bd_filter.cellular_classifier(
            np.ones(self.image[0].shape), np.ones((3, 4))
        )

        self.assertArrayAlmostEqual(
            np.array([False, True, False]), mask)
        self.assertAlmostEqual(2.494128145, cost)

    def test_cluster_colours(self):
        labels, centres = self.bd_filter._kmeans_cluster_colours(
            self.image)

        self.assertArrayAlmostEqual(
            np.arange(10), np.unique(labels)
        )
        self.assertEqual((10, 3), centres.shape)

    def test_BD_filter(self):
        mask_image = self.bd_filter.filter_image(self.image)

        self.assertEqual((200, 200), mask_image.shape)
