from unittest import TestCase
import numpy as np

from pyfibre.tools.extraction import (
    check_2D_arrays, distance_matrix, branch_angles
)


class TestFIRE(TestCase):

    def setUp(self):
        pass

    def test_FIRE(self):

        pos_2D = np.array([[1, 3],
                           [4, 2],
                           [1, 5]])

        indices = check_2D_arrays(pos_2D, pos_2D + 1.5, 2)

        self.assertEqual(indices[0], 2)
        self.assertEqual(indices[1], 0)

        answer_d_2D = np.array([[[0, 0], [3, -1], [0, 2]],
                                [[-3, 1], [0, 0], [-3, 3]],
                                [[0, -2], [3, -3], [0, 0]]])
        answer_r2_2D = np.array([[0, 10, 4],
                                [10, 0, 18],
                                [4, 18, 0]])
        d_2D, r2_2D = distance_matrix(pos_2D)

        self.assertAlmostEqual(abs(answer_d_2D - d_2D).sum(), 0, 7)
        self.assertAlmostEqual(abs(answer_r2_2D - r2_2D).sum(), 0, 7)

        direction = np.array([1, 0])
        vectors = d_2D[([2, 0], [0, 1])]
        r = np.sqrt(r2_2D[([2, 0], [0, 1])])

        answer_cos_the = np.array([0, 0.9486833])
        cos_the = branch_angles(direction, vectors, r)
        self.assertAlmostEqual(abs(answer_cos_the - cos_the).sum(), 0, 7)
