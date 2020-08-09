from unittest import TestCase

from pyfibre_shg_pl_trans.shg_pl_trans_factory import SHGPLTransFactory
from pyfibre_shg_pl_trans.shg_pl_trans_plugin import SHGPLTransPlugin


class TestSHGPLTransPlugin(TestCase):

    def setUp(self):

        self.plugin = SHGPLTransPlugin()

    def test_multi_image_factory(self):
        factory = self.plugin.multi_image_factories[0]
        self.assertIsInstance(factory, SHGPLTransFactory)
