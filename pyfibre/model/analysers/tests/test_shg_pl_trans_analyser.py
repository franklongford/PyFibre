import os
from tempfile import TemporaryDirectory

import numpy as np
from pandas import DataFrame, Series

from pyfibre.model.analysers.shg_pl_trans_analyser import (
    SHGPLTransAnalyser)
from pyfibre.model.pyfibre_workflow import PyFibreWorkflow
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.tests.probe_classes.shg_pl_trans_image import (
    ProbeSHGPLTransImage)
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, ProbeFibreSegment, ProbeCellSegment)
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_probe_graph)


class TestSHGPLTransAnalyser(PyFibreTestCase):

    def setUp(self):
        self.image, _, _, _ = generate_image()
        self.network = generate_probe_graph()
        self.fibre_networks = [ProbeFibreNetwork()]
        self.fibre_segments = [ProbeFibreSegment()]
        self.cell_segments = [ProbeCellSegment()]

        self.multi_image = ProbeSHGPLTransImage(
        )
        self.analyser = SHGPLTransAnalyser(
            multi_image=self.multi_image
        )
        self.workflow = PyFibreWorkflow()

    def test_save_load_segments(self):
        self.multi_image.shg_image = self.image
        self.multi_image.pl_image = np.zeros((10, 10))
        self.analyser._fibre_segments = self.fibre_segments
        self.analyser._cell_segments = self.cell_segments

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.analyser._save_segments()

            self.assertListEqual(
                ['test-shg-pl-trans_cell_segments.npy',
                 'test-shg-pl-trans_fibre_segments.npy'],
                os.listdir(f"{tmp_dir}/test-shg-pl-trans-pyfibre-analysis"
                           f"/data")
            )

            self.analyser._load_segments()

        self.assertEqual(1, len(self.analyser._fibre_segments))
        self.assertEqual(1, len(self.analyser._cell_segments))
        self.assertArrayAlmostEqual(
            self.fibre_segments[0].region.intensity_image,
            self.analyser._fibre_segments[0].region.intensity_image
        )
        self.assertArrayAlmostEqual(
            np.zeros((6, 4)),
            self.analyser._cell_segments[0].region.intensity_image
        )

    def test_create_metrics(self):
        self.analyser._fibre_networks = self.fibre_networks
        self.analyser._fibre_segments = self.fibre_segments
        self.analyser._cell_segments = self.cell_segments

        self.analyser.create_metrics(sigma=self.workflow.sigma)

        self.assertEqual(4, len(self.analyser._databases))
        self.assertEqual((30,), self.analyser._databases[0].shape)
        self.assertEqual((1, 11), self.analyser._databases[1].shape)
        self.assertEqual((1, 9), self.analyser._databases[2].shape)
        self.assertEqual((1, 11), self.analyser._databases[3].shape)

        self.assertIsInstance(self.analyser._databases[0], Series)
        for database in self.analyser._databases[1:]:
            self.assertIsInstance(database, DataFrame)

    def test_create_figures(self):

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.analyser.create_figures()

            figures = ['test-shg-pl-trans_PL.png',
                       'test-shg-pl-trans_trans.png']

            for figure in figures:
                self.assertIn(
                    figure,
                    os.listdir(f"{tmp_dir}/test-shg-pl-trans-pyfibre-analysis"
                               f"/fig")
                )
