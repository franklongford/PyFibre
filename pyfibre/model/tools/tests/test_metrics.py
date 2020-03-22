from unittest import TestCase

import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    fibre_metrics, fibre_network_metrics,
    nematic_tensor_metrics, region_shape_metrics,
    region_texture_metrics, network_metrics)
from pyfibre.tests.probe_classes import (
    ProbeFibre, ProbeFibreNetwork, generate_regions)


class TestAnalysis(TestCase):

    def setUp(self):

        self.regions = generate_regions()
        self.fibre_network = ProbeFibreNetwork()

    def test_nematic_tensor_metrics(self):

        metrics = nematic_tensor_metrics(
            self.regions[0], np.ones((10, 10, 2, 2)), 'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

    def test_region_shape_metrics(self):

        metrics = region_shape_metrics(
            self.regions[0], 'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(8, len(metrics))

    def test_region_texture_metrics(self):

        metrics = region_texture_metrics(
            self.regions[0], tag='test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(5, len(metrics))

        metrics = region_texture_metrics(
            self.regions[0], tag='test', glcm=True)

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(14, len(metrics))

        metrics = region_texture_metrics(
            self.regions[0], image=np.ones((10, 10)), tag='test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(5, len(metrics))

    def test_network_metrics(self):

        metrics = network_metrics(
            self.fibre_network.graph,
            self.fibre_network.red_graph,
            'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(4, len(metrics))

    def test_fibre_metricss(self):

        tot_fibres = [ProbeFibre(), ProbeFibre(), ProbeFibre()]

        fibre_database = fibre_metrics(tot_fibres)

        self.assertIsInstance(fibre_database, pd.DataFrame)
        self.assertEqual(3, len(fibre_database))
        self.assertEqual(3, len(fibre_database.columns))

    def test_fibre_network_analysis(self):

        self.fibre_network.fibres = self.fibre_network.generate_fibres()

        metrics = fibre_network_metrics(
            [self.fibre_network])

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual(15, len(metrics.columns))
