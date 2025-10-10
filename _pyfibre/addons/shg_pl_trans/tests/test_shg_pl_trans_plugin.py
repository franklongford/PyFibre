from unittest import TestCase

from ..shg_pl_trans_factory import SHGPLTransFactory
from ..shg_pl_trans_plugin import SHGPLTransPlugin


class TestSHGPLTransPlugin(TestCase):

    def setUp(self):

        self.plugin = SHGPLTransPlugin()

    def test_multi_image_factory(self):
        factory = self.plugin.multi_image_factories[0]
        self.assertIsInstance(factory, SHGPLTransFactory)
