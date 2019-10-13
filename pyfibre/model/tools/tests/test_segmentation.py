from unittest import TestCase

import networkx as nx
import numpy as np
from skimage.measure import regionprops

from pyfibre.model.tools.segment_utilities import (
    segments_to_binary, segment_check
)
from pyfibre.tests.probe_classes import (
    generate_image, generate_probe_network
)


class TestSegmentation(TestCase):

    def setUp(self):

        self.image, self.binary = generate_image()
        self.network = generate_probe_network()

