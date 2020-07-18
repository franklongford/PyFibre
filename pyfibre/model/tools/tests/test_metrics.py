from unittest import TestCase

import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    SHAPE_METRICS, TEXTURE_METRICS, FIBRE_METRICS,
    NETWORK_METRICS, STRUCTURE_METRICS,
    fibre_metrics, fibre_network_metrics,
    _segment_structure_tensor,
    structure_tensor_metrics, region_shape_metrics,
    region_texture_metrics, network_metrics,
    segment_metrics)
from pyfibre.tests.probe_classes.utilities import (
    generate_image, generate_regions)
from pyfibre.tests.probe_classes.objects import (
    ProbeSegment, ProbeFibre, ProbeFibreNetwork)


class TestAnalysis(TestCase):

    def setUp(self):

        self.regions = generate_regions()
        self.fibre_network = ProbeFibreNetwork()
        self.fibres = [ProbeFibre(), ProbeFibre(), ProbeFibre()]
        self.segment = ProbeSegment()
        self.segments = [self.segment]
        self.image, _, _, _ = generate_image()

    def test_structure_tensor_metrics(self):
        tensor_1d = np.array(
            [[[0, 1], [1, 0]],
             [[0, 0], [0, 1]]])

        metrics = structure_tensor_metrics(
            tensor_1d, 'test_1d')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        self.assertAlmostEqual(2.23606797, metrics['test_1d Anisotropy'])
        self.assertAlmostEqual(0.5, metrics['test_1d Local Anisotropy'])

        for metric in STRUCTURE_METRICS:
            self.assertIn(f'test_1d {metric}', metrics)

        tensor_2d = np.array(
            [[[[0, 1], [1, 0]],
              [[0, 0], [0, 1]]],
             [[[1, 0], [0, -1]],
              [[1, 0], [0, 0]]]])

        metrics = structure_tensor_metrics(
            tensor_2d, 'test_2d')

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        self.assertAlmostEqual(np.sqrt(2), metrics['test_2d Anisotropy'])
        self.assertAlmostEqual(0.5, metrics['test_2d Local Anisotropy'])

        for metric in STRUCTURE_METRICS:
            self.assertIn(f'test_2d {metric}', metrics)

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

    def test_segment_structure_tensor(self):

        segment = ProbeSegment()
        structure_tensor = np.ones((10, 10, 2, 2))

        segment_tensor = _segment_structure_tensor(
            segment, structure_tensor)

        self.assertEqual((9, 2, 2), segment_tensor.shape)

    def test_segment_metrics(self):

        database = segment_metrics(self.segments, self.image)
        self.assertEqual((1, 10), database.shape)

        metrics = STRUCTURE_METRICS + SHAPE_METRICS + TEXTURE_METRICS
        for metric in metrics:
            self.assertIn(f'Test Segment {metric}', database.columns)

        database = segment_metrics(
            self.segments, self.image, image_tag='Label')
        for metric in STRUCTURE_METRICS + TEXTURE_METRICS:
            self.assertIn(f'Test Segment Label {metric}', database.columns)
