import numpy as np
from skimage.io import imread

from pyfibre.model.tools.bd_cluster import (
    cluster_colours, cluster_mask,
    BDFilter
)
from pyfibre.tests.probe_classes import (
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

    def test_create_scaled_image(self):
        image_scaled = self.bd_filter._scale_image(self.image)

        self.assertEqual(image_scaled.shape, self.image.shape)
        self.assertAlmostEqual(130.66309166, image_scaled.mean())

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
        self.assertAlmostEqual(0.855886886, cost)

    def test_cluster_colours(self):
        labels, centres = cluster_colours(self.image)

        self.assertArrayAlmostEqual(
            np.arange(8), np.unique(labels)
        )
        self.assertEqual((8, 3), centres.shape)

    def test_BD_filter(self):
        mask_image = self.bd_filter.filter_image(self.image)

        self.assertEqual((200, 200), mask_image.shape)
