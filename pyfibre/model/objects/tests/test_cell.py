from unittest import TestCase

import numpy as np
from skimage.measure import regionprops

from pyfibre.tests.probe_classes import generate_image

from .. segments import CellSegment


class TestCell(TestCase):

    def setUp(self):
        (image, labels,
         binary, stack) = generate_image()
        region = regionprops(labels)[0]
        self.cell = CellSegment(
            image=image, region=region)

    def test_generate_database(self):

        database = self.cell.generate_database()
        self.assertEqual(22, len(database))

        image = np.ones((10, 10))
        image[2:, 2:] = 2

        database = self.cell.generate_database(image)
        self.assertEqual(22, len(database))

        self.cell.image = image

        database = self.cell.generate_database()
        self.assertEqual(22, len(database))

        self.cell.region = None
        with self.assertRaises(AttributeError):
            self.cell.generate_database()
