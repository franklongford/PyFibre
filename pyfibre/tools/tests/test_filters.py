from unittest import TestCase
import numpy as np
from pyfibre.tools.filters import gaussian, tubeness, hysteresis


class TestFilters(TestCase):

    def setUp(self):
        self.image = np.ones((5, 5))
        self.image[1, 1] = 5
        self.image[2, 2] = 10
        self.image[3, 3] = 8
        self.sigma = 1.0

    def test_gaussian(self):
        smoothed_image = gaussian(self.image, self.sigma)

        self.assertAlmostEqual(smoothed_image.mean(), 1.80, 6)
        self.assertAlmostEqual(smoothed_image.max(), 3.0771676, 6)
        self.assertAlmostEqual(smoothed_image.min(), 1.0455832, 6)

    def test_tubeness(self):
        tubeness_image = tubeness(self.image)

        self.assertAlmostEqual(tubeness_image.mean(), 0.53899511, 6)
        self.assertAlmostEqual(tubeness_image.max(), 1.1045664, 6)
        self.assertAlmostEqual(tubeness_image.min(), 0.3038492, 6)

    def test_hysteresis(self):
        hysteresis_image = hysteresis(self.image, alpha=2.0)

        self.assertTrue(hysteresis_image[1, 1])
        self.assertTrue(hysteresis_image[2, 2])
        self.assertTrue(hysteresis_image[3, 3])
