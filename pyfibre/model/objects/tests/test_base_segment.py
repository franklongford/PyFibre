from unittest import TestCase

import numpy as np
from skimage.measure import regionprops

from pyfibre.tests.probe_classes import generate_image

from .. segments import BaseSegment


class TestBaseSegment(TestCase):

    def setUp(self):
        (image, labels,
         binary, stack) = generate_image()
        region = regionprops(labels)[0]
        self.segment = BaseSegment(
            image=image, region=region)

    def test_generate_database(self):

        database = self.segment.generate_database()
        self.assertEqual(22, len(database))

        image = np.ones((10, 10))
        image[2:, 2:] = 2

        database = self.segment.generate_database(image)
        self.assertEqual(22, len(database))

        self.segment.image = image

        database = self.segment.generate_database()
        self.assertEqual(22, len(database))

        self.segment.region = None
        with self.assertRaises(AttributeError):
            self.segment.generate_database()
