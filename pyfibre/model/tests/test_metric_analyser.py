from unittest import TestCase

from pyfibre.tests.probe_classes import ProbeSHGPLTransImage

from .. metric_analyser import MetricAnalyser


class TestMetricAnalyser(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.metric_analyser = MetricAnalyser(
            filename='test', sigma=0.5
        )