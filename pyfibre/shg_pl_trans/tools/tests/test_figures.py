import os
from tempfile import NamedTemporaryFile

from pyfibre.tests.probe_classes.shg_image import ProbeSHGImage
from pyfibre.tests.probe_classes.shg_pl_trans_image import (
    ProbeSHGPLTransImage)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase

from pyfibre.shg_pl_trans_plugin.tools.figures import (
    create_shg_figures, create_shg_pl_trans_figures)


class TestFigures(PyFibreTestCase):

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
