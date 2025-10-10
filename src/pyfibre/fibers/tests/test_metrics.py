from unittest import TestCase
import pandas as pd

from pyfibre.fibers.fibre import Fibre
from pyfibre.fibers.fibre_network import FibreNetwork
from pyfibre.fibers.metrics import (
    FIBRE_METRICS,
    NETWORK_METRICS,
    fibre_metrics,
    fibre_network_metrics,
)
from pyfibre.testing.example_objects import (
    generate_image,
    generate_regions,
    generate_probe_graph,
)


class TestMetrics(TestCase):
    def setUp(self):
        self.regions = generate_regions()
        self.fibre_network = FibreNetwork(graph=generate_probe_graph())
        self.fibres = [Fibre(graph=generate_probe_graph()) for _ in range(3)]
        self.image, _, _, _ = generate_image()

    def test_fibre_metrics(self):
        metrics = fibre_metrics(self.fibres)

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual((3, 3), metrics.shape)

        for metric in FIBRE_METRICS:
            self.assertIn(f"Fibre {metric}", metrics)

    def test_fibre_network_analysis(self):
        metrics = fibre_network_metrics([self.fibre_network])

        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual((1, 8), metrics.shape)
        self.assertIn("No. Fibres", metrics)

        for metric in FIBRE_METRICS:
            self.assertIn(f"Mean Fibre {metric}", metrics)
            self.assertIsNotNone(metrics[f"Mean Fibre {metric}"])

        for metric in NETWORK_METRICS:
            self.assertIn(f"Fibre Network {metric}", metrics)
            self.assertIsNotNone(metrics[f"Fibre Network {metric}"])
