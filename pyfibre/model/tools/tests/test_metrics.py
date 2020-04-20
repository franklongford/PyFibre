from unittest import TestCase

import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    SHAPE_METRICS, TEXTURE_METRICS, FIBRE_METRICS,
    NETWORK_METRICS, NEMATIC_METRICS,
    fibre_metrics, fibre_network_metrics,
    nematic_tensor_metrics, region_shape_metrics,
    region_texture_metrics, network_metrics)
from pyfibre.tests.probe_classes.utilities import generate_regions
from pyfibre.tests.probe_classes.objects import (
    ProbeFibre, ProbeFibreNetwork)


class TestAnalysis(TestCase):

    def setUp(self):

        self.regions = generate_regions()
        self.fibre_network = ProbeFibreNetwork()
        self.fibres = [ProbeFibre(), ProbeFibre(), ProbeFibre()]

    def test_nematic_tensor_metrics(self):

        metrics = nematic_tensor_metrics(
            self.regions[0], np.ones((10, 10, 2, 2)), 'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        for metric in NEMATIC_METRICS:
            self.assertIn(f'test {metric}', metrics)

    def test_region_shape_metrics(self):

        metrics = region_shape_metrics(
            self.regions[0], 'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(4, len(metrics))

        for metric in SHAPE_METRICS:
            self.assertIn(f'test {metric}', metrics)

    def test_region_texture_metrics(self):

        metrics = region_texture_metrics(
            self.regions[0], tag='test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        for metric in TEXTURE_METRICS:
            self.assertIn(f'test {metric}', metrics)

        metrics = region_texture_metrics(
            self.regions[0], tag='test', glcm=True)

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(12, len(metrics))

        metrics = region_texture_metrics(
            self.regions[0], image=np.ones((10, 10)), tag='test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

    def test_fibre_metrics(self):

        metrics = fibre_metrics(self.fibres)

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual((3, 3), metrics.shape)

        for metric in FIBRE_METRICS:
            self.assertIn(f'Fibre {metric}', metrics)

    def test_network_metrics(self):

        metrics = network_metrics(
            self.fibre_network.graph,
            self.fibre_network.red_graph,
            3,
            'test')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(5, len(metrics))

        for metric in NETWORK_METRICS:
            self.assertIn(f'test Network {metric}', metrics)

    def test_fibre_network_analysis(self):

        self.fibre_network.fibres = self.fibres

        metrics = fibre_network_metrics(
            [self.fibre_network])

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual((1, 8), metrics.shape)
        self.assertIn('No. Fibres', metrics)

        for metric in FIBRE_METRICS:
            self.assertIn(f'Mean Fibre {metric}', metrics)

        for metric in NETWORK_METRICS:
            self.assertIn(f'Fibre Network {metric}', metrics)
