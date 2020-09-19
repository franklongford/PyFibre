from unittest import TestCase

from pyfibre.model.objects.segments import CellSegment, FibreSegment
from pyfibre.tests.probe_classes.utilities import generate_regions, generate_image
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork
from pyfibre.addons.shg_pl_trans.metric_analysers import (
    SHGMetricAnalyser, PLMetricAnalyser)


class TestMetricAnalyser(TestCase):

    def setUp(self):

        self.shg_metric_analyser = SHGMetricAnalyser(
            filename='test', sigma=0.5
        )
        self.pl_metric_analyser = PLMetricAnalyser(
            filename='test', sigma=0.5
        )
        self.fibre_network = ProbeFibreNetwork()

        self.image, _, _, _ = generate_image()
        regions = generate_regions()
        self.fibre_segments = [
            FibreSegment(region=region) for region in regions]
        self.cell_segments = [CellSegment(region=regions[1])]

        self.fibre_networks = [self.fibre_network]
        self.fibre_segment = self.fibre_segments[0]
        self.cell_segment = self.cell_segments[0]

    def test_analyse_shg(self):

        self.shg_metric_analyser.image = self.image
        self.shg_metric_analyser.segments = self.fibre_segments
        self.shg_metric_analyser.networks = self.fibre_networks
        (segment_metrics,
         network_metrics,
         global_metrics) = self.shg_metric_analyser.analyse()

        self.assertEqual((2, 11), segment_metrics.shape)
        self.assertIn('File', segment_metrics)
        self.assertEqual((1, 9), network_metrics.shape)
        self.assertIn('File', network_metrics)
        self.assertEqual((18,), global_metrics.shape)

        self.assertEqual(
            1, global_metrics['No. Fibres'])
        self.assertEqual(
            12/100, global_metrics['Fibre Segment Coverage'])

    def test_analyse_pl(self):

        self.pl_metric_analyser.image = self.image
        self.pl_metric_analyser.segments = self.cell_segments
        local_metrics, global_metrics = self.pl_metric_analyser.analyse()

        self.assertEqual((1, 11), local_metrics.shape)
        self.assertEqual((11,), global_metrics.shape)

        self.assertEqual(
            1, global_metrics['No. Cells'])
        self.assertEqual(
            3 / 100, global_metrics['Cell Segment Coverage'])
