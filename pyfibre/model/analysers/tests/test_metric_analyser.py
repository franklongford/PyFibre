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
        self.database = pd.DataFrame(
            {'one': np.array([1., 2., 3., 4.]),
             'two': np.array([4., 3., 2., 1.]),
             'weights': np.array([1, 1, 2, 1])}
        )
        self.metrics = ['one', 'two']

    def test_metric_averaging(self):
        averages = metric_averaging(
            self.database, self.metrics
        )
        self.assertAlmostEqual(2.5, averages['one'])
        self.assertAlmostEqual(2.5, averages['two'])

        averages = metric_averaging(
            self.database, self.metrics,
            weights=self.database['weights'].tolist()
        )
        self.assertAlmostEqual(2.6, averages['one'])
        self.assertAlmostEqual(2.4, averages['two'])

        averages = metric_averaging(
            self.database, self.metrics,
            weights=self.database['weights'].values
        )
        self.assertAlmostEqual(2.6, averages['one'])
        self.assertAlmostEqual(2.4, averages['two'])

    def test_metric_averaging_zero_values(self):
        self.database['one'] = np.array([1., 0, 3., 4.])
        averages = metric_averaging(
            self.database, self.metrics
        )
        self.assertAlmostEqual(2.0, averages['one'])
        self.assertAlmostEqual(2.5, averages['two'])

        self.database['two'] = np.array([4., 3., 0, 1.])
        averages = metric_averaging(
            self.database, self.metrics,
            weights=self.database['weights'].values
        )
        self.assertAlmostEqual(2.2, averages['one'])
        self.assertAlmostEqual(1.6, averages['two'])

    def test_metric_averaging_none_values(self):
        self.database['one'] = np.array([1., None, 3., 4.])
        averages = metric_averaging(
            self.database, self.metrics
        )
        self.assertAlmostEqual(8/3, averages['one'])
        self.assertAlmostEqual(2.5, averages['two'])

        self.database['weights'] = np.array([1., 2., None, 1.])
        averages = metric_averaging(
            self.database, self.metrics,
            weights=self.database['weights'].values
        )
        self.assertAlmostEqual(2.5, averages['one'])
        self.assertAlmostEqual(2.75, averages['two'])

    def test_metric_averaging_error(self):
        msg = ('Weights array must have same shape as '
               'database columns')
        with self.assertRaisesRegex(ValueError, msg):
            metric_averaging(
                self.database, self.metrics,
                weights=np.array([1, 2])
            )
