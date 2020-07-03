from unittest import TestCase

from pyfibre.model.objects.segments import CellSegment, FibreSegment
from pyfibre.tests.probe_classes.utilities import generate_regions
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork

from pyfibre.shg_pl_trans.tests.probe_classes import (
    ProbeSHGImage,
    ProbeSHGPLTransImage)
from pyfibre.shg_pl_trans.metric_analysers import (
    SHGMetricAnalyser, PLMetricAnalyser)


class TestMetricAnalyser(TestCase):

    def setUp(self):

        self.shg_multi_image = ProbeSHGImage()
        self.shg_pl_multi_image = ProbeSHGPLTransImage()
        self.shg_metric_analyser = SHGMetricAnalyser(
            filename='test', sigma=0.5
        )
        self.pl_metric_analyser = PLMetricAnalyser(
            filename='test', sigma=0.5
        )
        self.fibre_network = ProbeFibreNetwork()

        regions = generate_regions()
        self.fibre_segment = FibreSegment(region=regions[0])
        self.cell_segment = CellSegment(region=regions[1])

        self.fibre_networks = [self.fibre_network]
        self.fibre_segments = [self.fibre_segment]
        self.cell_segments = [self.cell_segment]

    def test_analyse_shg(self):

        self.shg_metric_analyser.image = self.shg_multi_image.shg_image
        self.shg_metric_analyser.segments = self.fibre_segments
        self.shg_metric_analyser.networks = self.fibre_networks
        (segment_metrics,
         network_metrics,
         global_metrics) = self.shg_metric_analyser.analyse()

        self.assertEqual((1, 11), segment_metrics.shape)
        self.assertIn('File', segment_metrics)
        self.assertEqual((1, 9), network_metrics.shape)
        self.assertIn('File', network_metrics)
        self.assertEqual((18,), global_metrics.shape)

    def test_analyse_pl(self):

        self.pl_metric_analyser.image = self.shg_pl_multi_image.pl_image
        self.pl_metric_analyser.segments = self.cell_segments
        local_metrics, global_metrics = self.pl_metric_analyser.analyse()

        self.assertEqual((1, 11), local_metrics.shape)
        self.assertEqual((11,), global_metrics.shape)
