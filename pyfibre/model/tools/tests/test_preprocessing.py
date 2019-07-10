from unittest import TestCase
import numpy as np

from pyfibre.model.tools.preprocessing import clip_intensities, nl_means


class TestPreprocessing(TestCase):

    def setUp(self):
        self.image = np.ones((5, 5))
        self.image[2, 2] = 10
        self.image[1, 1] = 5

    def test_clip_intensities(self):
        clipped_image = clip_intensities(self.image,
                                         p_intensity=(1, 95))

        self.assertEqual(clipped_image[2, 2], 1)
        self.assertEqual(clipped_image[1, 1], 1)

        clipped_image = clip_intensities(self.image,
                                         p_intensity=(1, 96))
        self.assertEqual(clipped_image[2, 2], 1)
        self.assertAlmostEqual(clipped_image[1, 1], 0.9523809, 6)

    def test_nl_means(self):

        denoised_image = nl_means(self.image)

        self.assertAlmostEqual(denoised_image.mean(), 1.5259023, 6)