import os
from unittest import TestCase, mock
from tempfile import NamedTemporaryFile

import pandas as pd

from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.tests.fixtures import test_shg_pl_trans_image_path

from ..pyfibre_app import PyFibreApplication


ITERATOR_PATH = 'pyfibre.cli.pyfibre_app.analysis_generator'


def dummy_iterate_images(dictionary, runner, analysers, reader):

    for key, value in dictionary.items():
        yield [
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object),
            pd.Series(dtype=object)
        ]


class TestPyFibreApplication(TestCase):

    def setUp(self):

        self.pyfibre_app = PyFibreApplication()

    def test_init(self):

        self.assertEqual(1, len(self.pyfibre_app.supported_readers))
        self.assertIsInstance(
            self.pyfibre_app.supported_readers['SHG-PL-Trans'],
            SHGPLTransReader)

    def test_init_pyfibre_workflow(self):

        workflow = self.pyfibre_app.runner

        self.assertEqual(0.5, workflow.sigma)
        self.assertEqual(0.5, workflow.alpha)
        self.assertFalse(workflow.ow_network)
        self.assertFalse(workflow.ow_segment)
        self.assertFalse(workflow.ow_metric)
        self.assertFalse(workflow.save_figures)

    def test_save_database(self):

        self.pyfibre_app.file_paths = [test_shg_pl_trans_image_path]

        with NamedTemporaryFile() as tmp_file:
            self.pyfibre_app.database_name = tmp_file.name

            with mock.patch(ITERATOR_PATH) as mock_iterate:
                mock_iterate.side_effect = dummy_iterate_images

                self.pyfibre_app.run()
                self.assertTrue(mock_iterate.called)

            self.assertTrue(
                os.path.exists(tmp_file.name + '.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_fibre.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_network.xls'))
            self.assertTrue(
                os.path.exists(tmp_file.name + '_cell.xls'))
