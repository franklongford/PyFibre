from unittest import TestCase

from pyfibre.tests.probe_classes.plugins import ProbePyFibrePlugin


class TestPyFibrePlugins(TestCase):

    def setUp(self):

        self.plugin = ProbePyFibrePlugin()

    def test_init(self):

        self.assertEqual('probe', self.plugin.name)
        self.assertEqual(0, self.plugin.version)
        self.assertEqual('pyfibre.plugin.probe.0', self.plugin.id)
        self.assertEqual(1, len(self.plugin.multi_image_factories))
