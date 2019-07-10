from unittest import TestCase
import numpy as np

from pyfibre.model.tools.extraction import (
    check_2D_arrays, distance_matrix, branch_angles
)


class TestFIRE(TestCase):

    def setUp(self):
        self.pos_2D = np.array([[1, 3], [4, 2], [1, 5]])
        self.answer_d_2D = np.array([[[0, 0], [3, -1], [0, 2]],
                                    [[-3, 1], [0, 0], [-3, 3]],
                                    [[0, -2], [3, -3], [0, 0]]])
        self.answer_r2_2D = np.array([[0, 10, 4],
                                      [10, 0, 18],
                                      [4, 18, 0]])
        self.answer_cos_the = np.array([0, 0.9486833])

    def test_check_2D_arrays(self):

        indices = check_2D_arrays(self.pos_2D,
                                  self.pos_2D + 1.5, 2)
        self.assertEqual(indices[0], 2)
        self.assertEqual(indices[1], 0)

    def test_distance_matrix(self):
        d_2D, r2_2D = distance_matrix(self.pos_2D)

        self.assertAlmostEqual(
            abs(self.answer_d_2D - d_2D).sum(), 0, 7)
        self.assertAlmostEqual(
            abs(self.answer_r2_2D - r2_2D).sum(), 0, 7)

    def test_branch_angles(self):

        direction = np.array([1, 0])
        vectors = self.answer_d_2D[([2, 0], [0, 1])]
        r = np.sqrt(self.answer_r2_2D[([2, 0], [0, 1])])

        cos_the = branch_angles(direction, vectors, r)
        self.assertAlmostEqual(
            abs(self.answer_cos_the - cos_the).sum(), 0, 7)
