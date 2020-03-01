from unittest import TestCase
import numpy as np

from pyfibre.model.tools.metrics import (
    fibre_metrics, fibre_network_metrics)
from pyfibre.tests.probe_classes import (
    ProbeFibre, ProbeFibreNetwork)


class TestAnalysis(TestCase):

    def test_fibre_analysis(self):

        tot_fibres = [ProbeFibre(), ProbeFibre(), ProbeFibre()]

        fibre_database = fibre_metrics(tot_fibres)
        self.assertEqual(3, len(fibre_database))
        self.assertEqual(11, len(fibre_database.columns))

    def test_fibre_network_analysis(self):

        fibre_network = ProbeFibreNetwork()
        fibre_network.fibres = fibre_network.generate_fibres()
        image = np.ones((10, 10))
        image[2:, 2:] = 2

        metrics = fibre_network_metrics(
            [fibre_network], image)

        self.assertEqual(18, len(metrics.columns))
