import os
from unittest import TestCase
from tempfile import TemporaryDirectory

from pyfibre.tests.probe_classes.analyser import ProbeAnalyser


class TestBaseAnalyser(TestCase):

    def setUp(self):
        self.analyser = ProbeAnalyser()

    def test_analysis_path(self):
        self.assertEqual(
            '/path/to/analysis/probe_multi_image-pyfibre-analysis',
            self.analyser.analysis_path)

    def test_make_directories(self):
        with TemporaryDirectory() as tmp_dir:
            self.analyser.multi_image.path = tmp_dir

            self.analyser.make_directories()

            self.assertTrue(
                os.path.exists(
                    f"{tmp_dir}/probe_multi_image-pyfibre-analysis"
                )
            )
