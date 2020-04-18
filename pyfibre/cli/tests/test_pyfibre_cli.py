import os
from unittest import TestCase, mock
from tempfile import NamedTemporaryFile

import pandas as pd

from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.model.image_analyser import ImageAnalyser
from pyfibre.tests.fixtures import test_shg_pl_trans_image_path

from ..pyfibre_cli import PyFibreCLI


ITERATOR_PATH = 'pyfibre.cli.pyfibre_cli.iterate_images'


def dummy_iterate_images(dictionary, analyser, reader):

    for key, value in dictionary.items():
        yield [
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object)
        ]


class TestPyFibreCLI(TestCase):

    def setUp(self):

        self.pyfibre_cli = PyFibreCLI()

    def test_init(self):

        self.assertIsInstance(
            self.pyfibre_cli.image_analyser, ImageAnalyser)
        self.assertEqual(1, len(self.pyfibre_cli.supported_readers))
        self.assertIsInstance(
            self.pyfibre_cli.supported_readers['SHG-PL-Trans'],
            SHGPLTransReader)

    def test_init_pyfibre_workflow(self):

        workflow = self.pyfibre_cli.image_analyser.workflow

        self.assertEqual(0.5, workflow.sigma)
        self.assertEqual(0.5, workflow.alpha)
        self.assertFalse(workflow.ow_network)
        self.assertFalse(workflow.ow_segment)
        self.assertFalse(workflow.ow_metric)
        self.assertFalse(workflow.save_figures)

    def test_save_database(self):

        with NamedTemporaryFile() as tmp_file:
            self.pyfibre_cli.database_name = tmp_file.name

            with mock.patch(ITERATOR_PATH) as mock_iterate:
                mock_iterate.side_effect = dummy_iterate_images

                self.pyfibre_cli.run(test_shg_pl_trans_image_path)
                self.assertTrue(mock_iterate.called)

            self.assertTrue(
                os.path.exists(tmp_file.name + '.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_network.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_cell.xls'))
