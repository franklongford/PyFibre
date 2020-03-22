from unittest import TestCase

from pyfibre.model.objects.segments import CellSegment, FibreSegment
from pyfibre.tests.probe_classes import (
    generate_regions,
    ProbeFibreNetwork,
    ProbeSHGImage,
    ProbeSHGPLTransImage
)

from .. metric_analyser import MetricAnalyser, generate_metrics


class TestMetricAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGImage()
        self.metric_analyser = MetricAnalyser(
            filename='test', sigma=0.5
        )
        self.fibre_network = ProbeFibreNetwork()
        self.fibre_network.fibres = self.fibre_network.generate_fibres()
        self.fibre_network.red_graph = self.fibre_network.generate_red_graph()

        regions = generate_regions()
        self.fibre_segment = FibreSegment(region=regions[0])
        self.cell_segment = CellSegment(region=regions[1])

        self.fibre_networks = [self.fibre_network]
        self.fibre_segments = [self.fibre_segment]
        self.cell_segments = [self.cell_segment]

    def test_analyse_shg(self):

        self.metric_analyser.image = self.multi_image.shg_image
        self.metric_analyser.segments = self.fibre_segments
        self.metric_analyser.networks = self.fibre_networks
        self.metric_analyser.analyse_shg()

    def test_generate_metrics(self):

        global_dataframe, local_dataframes = generate_metrics(
            self.multi_image,
            'test_shg',
            self.fibre_networks,
            self.fibre_segments,
            [],
            1.0
        )

        self.assertEqual(2, len(local_dataframes))
        self.assertEqual((1, 32), local_dataframes[0].shape)
        self.assertIsNone(local_dataframes[1])
        self.assertEqual((9,), global_dataframe.shape)

        global_dataframe, local_dataframes = generate_metrics(
            ProbeSHGPLTransImage(),
            'test_shg_pl',
            self.fibre_networks,
            self.fibre_segments,
            self.cell_segments,
            1.0
        )

        self.assertEqual(2, len(local_dataframes))
        self.assertEqual((1, 32), local_dataframes[0].shape)
        self.assertEqual((1, 17), local_dataframes[1].shape)
        self.assertEqual((16,), global_dataframe.shape)
