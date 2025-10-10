from unittest import TestCase

from pyfibre.tests.probe_classes.factories import (
    ProbeMultiImageFactory)
from pyfibre.tests.probe_classes.analyser import (
    ProbeAnalyser)
from pyfibre.tests.probe_classes.readers import (
    ProbeMultiImageReader)


class TestBaseMultiImageFactory(TestCase):

    def setUp(self):
        self.factory = ProbeMultiImageFactory()

    def test_init(self):
        self.assertEqual('Probe', self.factory.label)

    def test_create_reader(self):
        reader = self.factory.create_reader()
        self.assertIsInstance(reader, ProbeMultiImageReader)

    def test_create_analyser(self):
        analyser = self.factory.create_analyser()
        self.assertIsInstance(analyser, ProbeAnalyser)
