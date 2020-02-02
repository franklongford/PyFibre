from unittest import TestCase

from pyfibre.tests.probe_classes import (
    ProbeFibreNetwork,
    ProbeSHGPLTransImage
)

from .. metric_analyser import MetricAnalyser, generate_metrics


class TestMetricAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.metric_analyser = MetricAnalyser(
            filename='test', sigma=0.5
        )
        self.fibre_network = ProbeFibreNetwork()
        self.fibre_network.fibres = self.fibre_network.generate_fibres()
        self.fibre_network.red_graph = self.fibre_network.generate_red_graph()

        self.fibre_networks = [self.fibre_network]

    def test_generate_metrics(self):

        global_dataframe, local_dataframes = generate_metrics(
            self.multi_image,
            'test_python',
            self.fibre_networks,
            None,
            1.0,
            shg_analysis=True
        )

        self.assertEqual(2, len(local_dataframes))
        self.assertEqual((1, 19), local_dataframes[0].shape)
        self.assertEqual((11,), global_dataframe.shape)
