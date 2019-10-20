from unittest import TestCase

import networkx as nx
import numpy as np
from skimage.measure import regionprops

from pyfibre.io.tif_reader import load_image
from pyfibre.model.tools.bd_cluster import (
    prepare_composite_image
)
from pyfibre.model.tools.bd_cluster import (
    prepare_composite_image, cluster_colours,
    BD_filter
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_network
)
from pyfibre.tests.test_utilities import test_image_path


class TestBDCluster(TestCase):

    def setUp(self):
        self.image, self.labels, self.binary = generate_image()
        self.network = generate_probe_network()

        self.image_stack = load_image(test_image_path).mean(axis=-1)
        for image in self.image_stack:
            image /= image.max()

    def test_prepare_composite_image(self):
        pass

    def test_cluster_colours(self):
        pass

    def test_BD_filter(self):
        BD_filter(self.image_stack)
