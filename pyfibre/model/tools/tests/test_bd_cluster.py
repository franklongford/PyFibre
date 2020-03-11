from unittest import TestCase

import numpy as np
from skimage.io import imread

from pyfibre.model.tools.bd_cluster import (
    create_scaled_image, cluster_colours,
    BD_filter
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_graph
)
from pyfibre.tests.fixtures import test_image_path


class TestBDCluster(TestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()
        self.network = generate_probe_graph()

        self.image_stack = imread(test_image_path).mean(axis=-1)

        self.image = np.zeros(self.image_stack[0].shape + (3,))
        for index, image in enumerate(self.image_stack):
            self.image[..., index] = image / image.max()

    def test_create_scaled_image(self):
        image_scaled = create_scaled_image(self.image)

        self.assertEqual(image_scaled.shape, self.image.shape)
        self.assertAlmostEqual(131.05258333, image_scaled.mean())

    def test_cluster_colours(self):
        labels, centres = cluster_colours(self.image)

        self.assertListEqual(
            [0, 1, 2, 3, 4, 5, 6, 7], list(np.unique(labels))
        )
        self.assertEqual((8, 3), centres.shape)

    def test_BD_filter(self):
        mask_image = BD_filter(self.image)

        self.assertEqual((200, 200), mask_image.shape)
