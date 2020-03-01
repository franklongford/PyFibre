import numpy as np

from pyfibre.model.tools.filters import (
    gaussian, tubeness, hysteresis, derivatives,
    form_structure_tensor, form_nematic_tensor
)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestFilters(PyFibreTestCase):

    def setUp(self):
        self.image = np.ones((5, 5))
        self.image[1, 1] = 5
        self.image[2, 2] = 10
        self.image[3, 3] = 8
        self.sigma = 1.0

    def test_gaussian(self):
        smoothed_image = gaussian(self.image)

        self.assertArrayAlmostEqual(smoothed_image, smoothed_image)

        smoothed_image = gaussian(self.image, self.sigma)

        self.assertAlmostEqual(smoothed_image.mean(), 1.80, 6)
        self.assertAlmostEqual(smoothed_image.max(), 3.0771676, 6)
        self.assertAlmostEqual(smoothed_image.min(), 1.0455832, 6)

    def test_tubeness(self):
        tubeness_image = tubeness(self.image)

        self.assertAlmostEqual(tubeness_image.mean(), 0.53899511, 6)
        self.assertAlmostEqual(tubeness_image.max(), 1.1045664, 6)
        self.assertAlmostEqual(tubeness_image.min(), 0.3038492, 6)

        tubeness_image = tubeness(self.image, sigma_max=1)

        self.assertAlmostEqual(tubeness_image.mean(), 0.52744720, 6)
        self.assertAlmostEqual(tubeness_image.max(), 1.1045664, 6)
        self.assertAlmostEqual(tubeness_image.min(), 0.1719257, 6)

    def test_hysteresis(self):
        hysteresis_image = hysteresis(self.image, alpha=2.0)
        answer = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertArrayAlmostEqual(answer, hysteresis_image)

        hysteresis_image = hysteresis(self.image, alpha=0.1)

        self.assertArrayAlmostEqual(
            np.ones((5, 5)), hysteresis_image)

    def test_derivatives(self):
        first_derivatives = derivatives(self.image)

        dx = np.array([[0., 4., 0., 0., 0.],
                       [0., 0., 4.5, 0., 0.],
                       [0., -2., 0., 3.5, 0.],
                       [0., 0., -4.5, 0., 0.],
                       [0., 0., 0., -7., 0.]])
        dy = np.array([[0., 0., 0., 0., 0.],
                       [4., 0., -2, 0., 0.],
                       [0., 4.5, 0., -4.5, 0.],
                       [0., 0., 3.5, 0., -7.],
                       [0., 0., 0., 0., 0.]])

        self.assertArrayAlmostEqual(dx, first_derivatives[0])
        self.assertArrayAlmostEqual(dy, first_derivatives[1])

    def test_form_nematic_tensor(self):
        n_tensor = form_nematic_tensor(
            self.image, sigma=self.sigma)

        self.assertEqual(n_tensor.shape, (5, 5, 2, 2))

    def test_form_structure_tensor(self):
        j_tensor = form_structure_tensor(
            self.image, sigma=self.sigma)

        self.assertEqual(j_tensor.shape, (5, 5, 2, 2))
