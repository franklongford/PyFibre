import numpy as np

from pyfibre.tools.analysis import (
    tensor_analysis, angle_analysis
)
from pyfibre.testing.pyfibre_test_case import PyFibreTestCase


class TestAnalysis(PyFibreTestCase):

    def setUp(self):
        self.tensor = np.ones((3, 3, 2, 2))
        self.tensor[0, 0, 0, 1] = 2
        self.tensor[0, 0, 1, 0] = 2
        self.tensor[1, 1, 1] *= 4

    def test_tensor_analysis(self):
        tensor = np.array([[1, 0],
                           [0, 1]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor)
        self.assertArrayAlmostEqual(np.array([0]), tot_coher)
        self.assertArrayAlmostEqual(np.array([0]), tot_angle)
        self.assertArrayAlmostEqual(np.array([2]), tot_energy)

        tensor = np.array([[1, 0],
                           [0, -1]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor)
        self.assertArrayAlmostEqual(np.array([0]), tot_coher)
        self.assertArrayAlmostEqual(np.array([90]), tot_angle)
        self.assertArrayAlmostEqual(np.array([2]), tot_energy)

        tensor = np.array([[1, 0],
                           [0, 0]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor)
        self.assertArrayAlmostEqual(np.array([1]), tot_coher)
        self.assertArrayAlmostEqual(np.array([90]), tot_angle)
        self.assertArrayAlmostEqual(np.array([1]), tot_energy)

        tensor = np.array([[0, 0],
                           [0, 1]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor)
        self.assertArrayAlmostEqual(np.array([1]), tot_coher)
        self.assertArrayAlmostEqual(np.array([0]), tot_angle)
        self.assertArrayAlmostEqual(np.array([1]), tot_energy)

        tensor = np.array([[0, 1],
                           [1, 0]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor)
        self.assertArrayAlmostEqual(np.array([0]), tot_coher)
        self.assertArrayAlmostEqual(np.array([45]), tot_angle)
        self.assertArrayAlmostEqual(np.array([0]), tot_energy)

    def test_1d_tensor(self):
        tensor_1d = np.array(
            [[[0, 1], [1, 0]],
             [[0, 0], [0, 1]]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor_1d)
        self.assertArrayAlmostEqual(np.array([0, 1]), tot_coher)
        self.assertArrayAlmostEqual(np.array([45, 0]), tot_angle)
        self.assertArrayAlmostEqual(np.array([0, 1]), tot_energy)

    def test_2d_tensor(self):

        tensor_2d = np.array(
            [[[[0, 1], [1, 0]],
              [[0, 0], [0, 1]]],
             [[[1, 0], [0, -1]],
              [[1, 0], [0, 0]]]])

        tot_coher, tot_angle, tot_energy = tensor_analysis(tensor_2d)
        self.assertArrayAlmostEqual(np.array([[0, 1], [0, 1]]), tot_coher)
        self.assertArrayAlmostEqual(np.array([[45, 0], [90, 90]]), tot_angle)
        self.assertArrayAlmostEqual(np.array([[0, 1], [2, 1]]), tot_energy)

    def test_angle_analysis(self):
        angles = np.array([45, 90, 100,
                           180, 45, 45,
                           45, 90, 180])
        angle_sdi, angle_x = angle_analysis(angles)

        self.assertAlmostEqual(angle_sdi, 0.01125)
        self.assertEqual((201,), angle_x.shape)

        angle_sdi, angle_x = angle_analysis(angles, n_bin=10)

        self.assertEqual(angle_sdi, 0.225)
        self.assertEqual((11,), angle_x.shape)

        weights = np.arange(angles.size)
        angle_sdi, angle_x = angle_analysis(angles, weights=weights)

        self.assertAlmostEqual(angle_sdi, 0.012)
        self.assertEqual((201,), angle_x.shape)
