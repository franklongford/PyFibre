import numpy as np

from pyfibre.model.tools.analysis import (
    tensor_analysis, angle_analysis
)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestAnalysis(PyFibreTestCase):

    def setUp(self):
        self.tensor = np.ones((3, 3, 2, 2))
        self.tensor[0, 1, 0] *= 2
        self.tensor[1, 1, 1] *= 4

    def test_tensor_analysis(self):

        tot_anis_answer = np.array([[1.0, 0.74535599, 1.0],
                                    [1.0, 1.70880074, 1.0],
                                    [1.0, 1.0, 1.0]])
        tot_angle_answer = np.array([[45, 58.2825255, 45],
                                     [45, 34.7219773, 45],
                                     [45, 45, 45]])
        tot_energy_answer = np.array([[2, 3, 2],
                                      [2, 5, 2],
                                      [2, 2, 2]])

        tot_anis, tot_angle, tot_energy = tensor_analysis(self.tensor)

        self.assertArrayAlmostEqual(tot_anis_answer, tot_anis)
        self.assertArrayAlmostEqual(tot_angle_answer, tot_angle)
        self.assertArrayAlmostEqual(tot_energy_answer, tot_energy)

    def test_angle_analysis(self):
        angles = np.array([45, 90, 100,
                           180, 45, 45,
                           45, 90, 180])
        weights = np.ones(angles.shape)
        angle_sdi, _ = angle_analysis(angles, weights, n_bin=10)

        self.assertEqual(angle_sdi, 0.225)
