import os
from tempfile import TemporaryDirectory

from pandas import DataFrame, Series

from pyfibre.model.analysers.shg_analyser import SHGAnalyser
from pyfibre.model.pyfibre_workflow import PyFibreWorkflow
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, ProbeFibreSegment, ProbeCellSegment)
from pyfibre.tests.probe_classes.shg_image import ProbeSHGImage
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_probe_graph)


class TestSHGAnalyser(PyFibreTestCase):

    def setUp(self):
        self.image, _, _, _ = generate_image()
        self.network = generate_probe_graph()
        self.fibre_networks = [ProbeFibreNetwork()]
        self.fibre_segments = [ProbeFibreSegment()]
        self.cell_segments = [ProbeCellSegment()]

        self.multi_image = ProbeSHGImage()
        self.analyser = SHGAnalyser(
            multi_image=self.multi_image
        )
        self.workflow = PyFibreWorkflow()

    def test_file_paths(self):
        self.multi_image.path = '/path/to/image'
        self.assertEqual(
            '/path/to/image/test-shg-pyfibre-analysis/data',
            self.analyser.data_path)
        self.assertEqual(
            '/path/to/image/test-shg-pyfibre-analysis/fig',
            self.analyser.fig_path)
        self.assertEqual(
            '/path/to/image/test-shg-pyfibre-analysis/data'
            '/test-shg',
            self.analyser._data_file)
        self.assertEqual(
            '/path/to/image/test-shg-pyfibre-analysis/fig'
            '/test-shg',
            self.analyser._fig_file)

    def test_get_ow_options(self):

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir

            ow_network, ow_segment, ow_metric = (
                self.analyser.get_analysis_options(self.workflow)
            )

            self.assertTrue(ow_network)
            self.assertTrue(ow_segment)
            self.assertTrue(ow_metric)

    def test_make_directories(self):
        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.assertTrue(
                os.path.exists(
                    f"{tmp_dir}/test-shg-pyfibre-analysis"
                )
            )
            self.assertTrue(
                os.path.exists(
                    f"{tmp_dir}/test-shg-pyfibre-analysis/data"
                )
            )
            self.assertTrue(
                os.path.exists(
                    f"{tmp_dir}/test-shg-pyfibre-analysis/fig"
                )
            )

    def test_save_load_networks(self):
        self.analyser._network = self.network
        self.analyser._fibre_networks = self.fibre_networks

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.analyser._save_networks()

            self.assertListEqual(
                ['test-shg_fibre_networks.json',
                 'test-shg_network.pkl'],
                os.listdir(f"{tmp_dir}/test-shg-pyfibre-analysis/data")
            )

            self.analyser._load_networks()

        self.assertListEqual(
            list(self.network.nodes),
            list(self.analyser._network.nodes))
        self.assertListEqual(
            list(self.network.edges),
            list(self.analyser._network.edges))

        self.assertEqual(1, len(self.analyser._fibre_networks))
        self.assertListEqual(
            list(self.fibre_networks[0].graph.nodes),
            list(self.analyser._fibre_networks[0].graph.nodes))
        self.assertListEqual(
            list(self.fibre_networks[0].graph.edges),
            list(self.analyser._fibre_networks[0].graph.edges))

    def test_save_load_segments(self):
        self.multi_image.shg_image = self.image
        self.analyser._fibre_segments = self.fibre_segments
        self.analyser._cell_segments = self.cell_segments

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.analyser._save_segments()

            self.assertListEqual(
                ['test-shg_fibre_segments.npy',
                 'test-shg_cell_segments.npy'],
                os.listdir(f"{tmp_dir}/test-shg-pyfibre-analysis/data")
            )

            self.analyser._load_segments()

        self.assertEqual(1, len(self.analyser._fibre_segments))
        self.assertEqual(1, len(self.analyser._cell_segments))
        self.assertArrayAlmostEqual(
            self.fibre_segments[0].region.intensity_image,
            self.analyser._fibre_segments[0].region.intensity_image
        )
        self.assertArrayAlmostEqual(
            self.cell_segments[0].region.intensity_image,
            self.analyser._cell_segments[0].region.intensity_image
        )

    def test_save_load_databases(self):
        self.analyser._databases = tuple(
            [DataFrame()] * 4
        )

        with TemporaryDirectory() as tmp_dir:
            self.multi_image.path = tmp_dir
            self.analyser.make_directories()

            self.analyser._save_databases()

            self.assertListEqual(
                ['test-shg_network_metric.xls',
                 'test-shg_network_metric.h5',
                 'test-shg_fibre_metric.xls',
                 'test-shg_global_metric.xls',
                 'test-shg_fibre_metric.h5',
                 'test-shg_cell_metric.h5',
                 'test-shg_global_metric.h5',
                 'test-shg_cell_metric.xls'],
                os.listdir(f"{tmp_dir}/test-shg-pyfibre-analysis/data")
            )

            self.analyser._load_databases()

        self.assertEqual(4, len(self.analyser._databases))

    def test_network_analysis(self):

        self.multi_image.shg_image = self.multi_image.shg_image[:50, :50]
        self.multi_image.preprocess_images()

        self.analyser.network_analysis(
            sigma=self.workflow.sigma,
            alpha=self.workflow.alpha,
            scale=self.workflow.scale,
            p_denoise=self.workflow.p_denoise,
            fire_parameters=self.workflow.fire_parameters
        )

        self.assertEqual(38, self.analyser._network.number_of_nodes())
        self.assertEqual(37, self.analyser._network.number_of_edges())
        self.assertEqual(2, len(self.analyser._fibre_networks))

    def test_create_metrics(self):
        self.analyser._fibre_networks = self.fibre_networks
        self.analyser._fibre_segments = self.fibre_segments
        self.analyser._cell_segments = self.cell_segments

        self.analyser.create_metrics(sigma=self.workflow.sigma)

        self.assertEqual((19,), self.analyser._databases[0].shape)
        self.assertEqual((1, 11), self.analyser._databases[1].shape)
        self.assertEqual((1, 9), self.analyser._databases[2].shape)

        self.assertIsInstance(self.analyser._databases[0], Series)
        for database in self.analyser._databases[1:3]:
            self.assertIsInstance(database, DataFrame)
        self.assertIsNone(self.analyser._databases[3])
