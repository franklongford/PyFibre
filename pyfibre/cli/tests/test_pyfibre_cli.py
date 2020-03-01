from unittest import TestCase

from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.model.image_analyser import ImageAnalyser

from ..pyfibre_cli import PyFibreCLI


class TestPyFibreCLI(TestCase):

    def setUp(self):

        self.pyfibre_cli = PyFibreCLI()

    def test_init(self):

        self.assertIsInstance(
            self.pyfibre_cli.image_analyser, ImageAnalyser)
        self.assertIsInstance(
            self.pyfibre_cli.reader, SHGPLTransReader)
