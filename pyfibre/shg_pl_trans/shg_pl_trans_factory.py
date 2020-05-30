from pyfibre.core.base_multi_image_factory import BaseMultiImageFactory

from .shg_pl_trans_analyser import SHGPLTransAnalyser
from pyfibre.shg_pl_trans.shg_pl_trans_reader import SHGPLTransReader


class SHGPLTransFactory(BaseMultiImageFactory):

    def get_label(self):
        return 'SHG-PL-Trans'

    def get_reader(self):
        """Returns list of BaseMultiImageReader classes able to load
        the BaseMultiImage class created by this factory"""
        return SHGPLTransReader

    def get_analyser(self):
        """Returns list of BaseMultiImageAnalyser classes able to analyse
        the BaseMultiImage class created by this factory"""
        return SHGPLTransAnalyser
