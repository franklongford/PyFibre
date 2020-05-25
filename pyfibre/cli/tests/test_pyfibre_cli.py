import os
from unittest import TestCase, mock
from tempfile import NamedTemporaryFile

import pandas as pd

from pyfibre.cli.pyfibre_cli import PyFibreApplication
from pyfibre.core.core_pyfibre_plugin import CorePyFibrePlugin
from pyfibre.tests.probe_classes.plugins import ProbePyFibrePlugin


ITERATOR_PATH = 'pyfibre.cli.pyfibre_cli.PyFibreRunner.run'


def dummy_iterate_images(dictionary, analyser, reader):
    for key, value in dictionary.items():
        yield [pd.Series(dtype=object)] * len(analyser.database_names)


class TestPyFibreApplication(TestCase):

    def setUp(self):
        plugins = [CorePyFibrePlugin(), ProbePyFibrePlugin()]
        self.pyfibre_app = PyFibreApplication(
            plugins=plugins
        )

    def test_init(self):

        workflow = self.pyfibre_app.runner

        self.assertEqual(0.5, workflow.sigma)
        self.assertEqual(0.5, workflow.alpha)
        self.assertFalse(workflow.ow_network)
        self.assertFalse(workflow.ow_segment)
        self.assertFalse(workflow.ow_metric)
        self.assertFalse(workflow.save_figures)

        self.assertEqual(1, len(self.pyfibre_app.supported_analysers))
        self.assertEqual(1, len(self.pyfibre_app.supported_readers))

    def test_run(self):

        self.pyfibre_app.file_paths = ['/path/to/some/image']

        with NamedTemporaryFile() as tmp_file:
            self.pyfibre_app.database_name = tmp_file.name

            with mock.patch(ITERATOR_PATH) as mock_iterate:
                mock_iterate.side_effect = dummy_iterate_images

                self.pyfibre_app.run()
                self.assertTrue(mock_iterate.called)

            self.assertTrue(
                os.path.exists(tmp_file.name + '_probe.xls'))
