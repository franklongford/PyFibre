from unittest import TestCase
import numpy as np

from pyfibre.model.tools.analysis import (
    tensor_analysis, angle_analysis
)


class TestAnalysis(TestCase):

    def setUp(self):
        self.tensor = np.ones((3, 3, 2, 2))
        self.tensor[0, 1, 0] *= 2
        self.tensor[1, 1, 1] *= 4

    def test_tensor_analysis(self):
        tot_anis, tot_angle, tot_energy = tensor_analysis(self.tensor)

        self.assertAlmostEqual(tot_anis[1, 1], 1.70880074, 6)
        self.assertAlmostEqual(tot_anis[0, 1], 0.74535599, 6)

        self.assertAlmostEqual(tot_angle[1, 1], 34.7219773, 6)
        self.assertAlmostEqual(tot_angle[0, 1], 58.2825255, 6)

        self.assertEqual(tot_energy[1, 1], 5.0)
        self.assertEqual(tot_energy[0, 1], 3.0)

    def test_angle_analysis(self):
        angles = np.array([45, 90, 100,
                           180, 45, 45,
                           45, 90, 180])
        weights = np.ones(angles.shape)
        angle_sdi, _ = angle_analysis(angles, weights, n_bin=10)

        self.assertEqual(angle_sdi, 0.225)
