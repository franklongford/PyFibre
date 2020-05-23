from pyfibre.model.core.base_pyfibre_plugin import BasePyFibrePlugin

from .shg_pl_trans_factory import SHGPLTransFactory


class SHGPLTransPlugin(BasePyFibrePlugin):

    def get_multi_image_factories(self):
        return [SHGPLTransFactory]
