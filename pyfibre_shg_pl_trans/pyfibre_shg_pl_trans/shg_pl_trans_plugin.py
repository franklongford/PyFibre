from pyfibre.core.base_pyfibre_plugin import BasePyFibrePlugin

from .shg_pl_trans_factory import SHGPLTransFactory
from .tests.fixtures import test_shg_pl_trans_image_path


class SHGPLTransPlugin(BasePyFibrePlugin):
    """Plugin that contributes a factory that can analyse
    an image format made of SHG and PL signals."""

    def get_name(self):
        return 'pyfibre_shg_pl_trans'

    def get_version(self):
        return 0

    def get_test_files(self):
        return [test_shg_pl_trans_image_path]

    def get_multi_image_factories(self):
        return [SHGPLTransFactory]
