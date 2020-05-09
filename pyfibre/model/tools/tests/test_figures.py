import os
from tempfile import NamedTemporaryFile

import numpy as np
from skimage.measure import regionprops

from pyfibre.tests.probe_classes.utilities import generate_image
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork
from pyfibre.tests.probe_classes.shg_image import ProbeSHGImage
from pyfibre.tests.probe_classes.shg_pl_trans_image import (
    ProbeSHGPLTransImage)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase

from ..figures import (
    create_figure,
    create_hsb_image,
    create_tensor_image,
    create_region_image,
    create_network_image,
    create_shg_figures,
    create_shg_pl_trans_figures
)


class TestFigures(PyFibreTestCase):

    def setUp(self):

        (self.image, labels,
         self.binary, _) = generate_image()
        self.segments = regionprops(labels)
        self.fibre_network = ProbeFibreNetwork()

    def test_create_figure(self):

        with NamedTemporaryFile() as temp_file:
            create_figure(self.image, temp_file.name)

            self.assertTrue(
                os.path.exists(temp_file.name + '.png')
            )

    def test_create_hsb_image(self):
        hsb_image = create_hsb_image(
            self.image, np.ones((10, 10)))

        self.assertEqual(
            (10, 10, 3), hsb_image.shape)
        self.assertArrayAlmostEqual(
            np.ones((10, 10)), hsb_image[..., 0])
        self.assertArrayAlmostEqual(
            np.zeros((10, 10)), hsb_image[..., 1])
        self.assertArrayAlmostEqual(
            np.zeros((10, 10)), hsb_image[..., 2])

    def test_create_tensor_image(self):

        tensor_image = create_tensor_image(self.image)

        self.assertEqual(
            (10, 10, 3), tensor_image.shape)

    def test_create_segment_image(self):
        segment_image = create_region_image(
            self.image, self.segments
        )

        indices = np.where(self.binary == 0)

        self.assertEqual(
            (10, 10, 3), segment_image.shape)
        self.assertTrue(
            np.allclose(segment_image[..., 0][indices], 0.0075)
        )
        self.assertTrue(
            np.allclose(segment_image[..., 1][indices], 0.0075)
        )
        self.assertTrue(
            np.allclose(segment_image[..., 2][indices], 0.0075)
        )

    def test_create_network_image(self):

        image = create_network_image(
            self.image,
            [self.fibre_network.graph])

        # Expect an RGB image back
        self.assertEqual(3, image.ndim)

    def test_create_shg_figures(self):
        multi_image = ProbeSHGImage()

        with NamedTemporaryFile() as tmp_file:
            create_shg_figures(multi_image, tmp_file.name)

            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_SHG.png"))
            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_tensor.png"))

    def test_create_shg_pl_trans_figures(self):
        multi_image = ProbeSHGPLTransImage()

        with NamedTemporaryFile() as tmp_file:
            create_shg_pl_trans_figures(multi_image, tmp_file.name)

            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_SHG.png"))
            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_tensor.png"))
            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_PL.png"))
            self.assertTrue(
                os.path.exists(f"{tmp_file.name}_trans.png"))
