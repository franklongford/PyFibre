from unittest import TestCase

import numpy as np
import pandas as pd

from pyfibre.segments.metrics import (
    SHAPE_METRICS,
    TEXTURE_METRICS,
    STRUCTURE_METRICS,
    _region_sample,
    structure_tensor_metrics,
    region_shape_metrics,
    region_texture_metrics,
    segment_metrics,
)
from pyfibre.segments.segment import Segment
from pyfibre.testing.example_objects import generate_image, generate_regions


class TestMetrics(TestCase):
    def setUp(self):
        self.regions = generate_regions()
        self.segment = Segment(name="Test", region=self.regions[0])
        self.segments = [self.segment]
        self.image, _, _, _ = generate_image()

    def test_region_sample(self):
        structure_tensor = np.ones((10, 10, 2, 2))

        region_tensor = _region_sample(self.regions[0], structure_tensor)

        self.assertEqual((9, 2, 2), region_tensor.shape)

    def test_structure_tensor_metrics(self):
        tensor_1d = np.array([[[0, 1], [1, 0]], [[0, 0], [0, 1]]])

        metrics = structure_tensor_metrics(tensor_1d, "test_1d")

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        self.assertAlmostEqual(5.0, metrics["test_1d Coherence"])
        self.assertAlmostEqual(0.5, metrics["test_1d Local Coherence"])

        for metric in STRUCTURE_METRICS:
            self.assertIn(f"test_1d {metric}", metrics)

        tensor_2d = np.array(
            [
                [[[0, 1], [1, 0]], [[0, 0], [0, 1]]],
                [[[1, 0], [0, -1]], [[1, 0], [0, 0]]],
            ]
        )

        metrics = structure_tensor_metrics(tensor_2d, "test_2d")

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        self.assertAlmostEqual(2, metrics["test_2d Coherence"])
        self.assertAlmostEqual(0.5, metrics["test_2d Local Coherence"])

        for metric in STRUCTURE_METRICS:
            self.assertIn(f"test_2d {metric}", metrics)

    def test_region_shape_metrics(self):
        metrics = region_shape_metrics(self.regions[0], "test")

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(4, len(metrics))

        for metric in SHAPE_METRICS:
            self.assertIn(f"test {metric}", metrics)

        self.assertAlmostEqual(9, metrics["test Area"])
        self.assertAlmostEqual(1.77245385, metrics["test Circularity"])
        self.assertAlmostEqual(0.69584728, metrics["test Eccentricity"])
        self.assertAlmostEqual(0.375, metrics["test Coverage"])

    def test_region_texture_metrics(self):
        metrics = region_texture_metrics(self.regions[0], tag="test")

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        for metric in TEXTURE_METRICS:
            self.assertIn(f"test {metric}", metrics)

        self.assertAlmostEqual(3.55555555, metrics["test Mean"])
        self.assertAlmostEqual(1.83249138, metrics["test STD"])
        self.assertAlmostEqual(1.35164411, metrics["test Entropy"])

        metrics = region_texture_metrics(self.regions[0], tag="test", glcm=True)

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(12, len(metrics))

        metrics = region_texture_metrics(
            self.regions[0], image=np.ones((10, 10)), tag="test"
        )

        self.assertIsInstance(metrics, pd.Series)
        self.assertEqual(3, len(metrics))

        self.assertAlmostEqual(1, metrics["test Mean"])
        self.assertAlmostEqual(0, metrics["test STD"])
        self.assertAlmostEqual(0, metrics["test Entropy"])

    def test_segment_metrics(self):
        database = segment_metrics(self.segments, self.image)
        self.assertEqual((1, 10), database.shape)

        metrics = STRUCTURE_METRICS + SHAPE_METRICS + TEXTURE_METRICS
        for metric in metrics:
            self.assertIn(f"Test Segment {metric}", database.columns)

        database = segment_metrics(self.segments, self.image, image_tag="Label")
        for metric in STRUCTURE_METRICS + TEXTURE_METRICS:
            self.assertIn(f"Test Segment Label {metric}", database.columns)
