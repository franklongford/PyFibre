from unittest import TestCase

import numpy as np
import pandas as pd

from pyfibre.model.analysers.metric_analyser import (
    MetricAnalyser, metric_averaging)


class ProbeMetricAnalyser(MetricAnalyser):

    def analyse(self):
        return [pd.DataFrame()]


class TestMetricAnalyser(TestCase):

    def setUp(self):
        self.metric_analyser = ProbeMetricAnalyser()

    def test_metric_averaging(self):
        database = pd.DataFrame(
            {'one': np.array([1., 2., 3., 4.]),
             'two': np.array([4., 3., 2., 1.])}
        )
        metrics = ['one']

        averages = metric_averaging(
            database, metrics
        )
        self.assertAlmostEqual(2.5, averages['one'])

        averages = metric_averaging(
            database, metrics, weights=database['two'].values
        )
        self.assertAlmostEqual(2.0, averages['one'])

        database['one'][1] = None
        averages = metric_averaging(
            database, metrics
        )
        self.assertAlmostEqual(8/3, averages['one'])

        database['two'][2] = None
        averages = metric_averaging(
            database, metrics, weights=database['two'].values
        )
        self.assertAlmostEqual(1.6, averages['one'])
