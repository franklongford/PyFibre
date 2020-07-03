from unittest import TestCase
import pandas as pd

from pyfibre.model.analysers.metric_analyser import MetricAnalyser


class ProbeMetricAnalyser(MetricAnalyser):

    def analyse(self):
        return [pd.Dataframe()]


class TestMetricAnalyser(TestCase):

    def setUp(self):
        self.metric_analyser = ProbeMetricAnalyser()
