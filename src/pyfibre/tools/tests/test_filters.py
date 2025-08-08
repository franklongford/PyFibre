import numpy as np
from unittest import TestCase

from pyfibre.tools.filters import (
    gaussian,
    tubeness,
    hysteresis,
    derivatives,
    form_structure_tensor,
    form_nematic_tensor,
)


class TestFilters(TestCase):
    def setUp(self):
        self.image = np.ones((5, 5))
        self.image[1, 1] = 5
        self.image[2, 2] = 10
        self.image[3, 3] = 8
        self.sigma = 1.0

    def test_gaussian(self):
        smoothed_image = gaussian(self.image)

        np.testing.assert_almost_equal(smoothed_image, smoothed_image)

        smoothed_image = gaussian(self.image, self.sigma)

        self.assertAlmostEqual(1.80, smoothed_image.mean(), 6)
        self.assertAlmostEqual(3.0771676, smoothed_image.max(), 6)
        self.assertAlmostEqual(1.0455832, smoothed_image.min(), 6)

    def test_tubeness(self):
        tubeness_image = tubeness(self.image)
        tubeness_image /= tubeness_image.max()

        self.assertAlmostEqual(0.3403651, tubeness_image.mean(), 6)
        self.assertAlmostEqual(0.0930330, tubeness_image.min(), 6)

        tubeness_image = tubeness(self.image, sigma_max=1)
        tubeness_image /= tubeness_image.max()

        self.assertAlmostEqual(0.3136189, tubeness_image.mean(), 6)
        self.assertAlmostEqual(0.0594625, tubeness_image.min(), 6)

    def test_hysteresis(self):
        hysteresis_image = hysteresis(self.image, alpha=2.0)
        answer = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_almost_equal(answer, hysteresis_image)

        hysteresis_image = hysteresis(self.image, alpha=0.1)

        np.testing.assert_almost_equal(np.ones((5, 5)), hysteresis_image)

    def test_derivatives(self):
        first_derivatives = derivatives(self.image)

        dx = np.array(
            [
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.5, 0.0, 0.0],
                [0.0, -2.0, 0.0, 3.5, 0.0],
                [0.0, 0.0, -4.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, -7.0, 0.0],
            ]
        )
        dy = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, -2, 0.0, 0.0],
                [0.0, 4.5, 0.0, -4.5, 0.0],
                [0.0, 0.0, 3.5, 0.0, -7.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual((2, 5, 5), first_derivatives.shape)
        np.testing.assert_almost_equal(dx, first_derivatives[0])
        np.testing.assert_almost_equal(dy, first_derivatives[1])

        second_derivatives = derivatives(self.image, rank=2)

        ddx = np.array(
            [
                [0.0, -4.0, 4.5, 0.0, 0.0],
                [0.0, -3.0, 0, 1.75, 0.0],
                [0.0, 0.0, -4.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, -5.25, 0.0],
                [0.0, 0.0, 4.5, -7.0, 0.0],
            ]
        )
        dxdy = np.array(
            [
                [4.0, 0.0, -2.0, 0.0, 0.0],
                [0.0, 2.25, 0.0, -2.25, 0.0],
                [-2.0, 0.0, 2.75, 0.0, -3.5],
                [0.0, -2.25, 0.0, 2.25, 0.0],
                [0.0, 0.0, -3.5, 0.0, 7.0],
            ]
        )
        ddy = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [-4.0, -3.0, 0, 1.0, 0.0],
                [4.5, 0, -4.5, 0, 4.5],
                [0.0, 1.75, 0.0, -5.25, -7.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual((4, 5, 5), second_derivatives.shape)
        np.testing.assert_almost_equal(ddx, second_derivatives[0])
        np.testing.assert_almost_equal(dxdy, second_derivatives[1])
        np.testing.assert_almost_equal(dxdy, second_derivatives[2])
        np.testing.assert_almost_equal(ddy, second_derivatives[3])

    def test_form_nematic_tensor(self):
        n_tensor = form_nematic_tensor(self.image)

        self.assertEqual((5, 5, 2, 2), n_tensor.shape)

        n_tensor = form_nematic_tensor(self.image, sigma=self.sigma)

        self.assertEqual((5, 5, 2, 2), n_tensor.shape)

        n_tensor = form_nematic_tensor(
            np.array([self.image, self.image]), sigma=self.sigma
        )

        self.assertEqual((2, 5, 5, 2, 2), n_tensor.shape)

    def test_form_structure_tensor(self):
        j_tensor = form_structure_tensor(self.image, sigma=self.sigma)

        self.assertEqual((5, 5, 2, 2), j_tensor.shape)

        j_tensor = form_structure_tensor(
            np.array([self.image, self.image]), sigma=self.sigma
        )

        self.assertEqual((2, 5, 5, 2, 2), j_tensor.shape)
