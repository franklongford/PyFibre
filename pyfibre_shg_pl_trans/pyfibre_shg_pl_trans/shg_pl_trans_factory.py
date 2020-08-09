from pyfibre.core.base_multi_image_factory import BaseMultiImageFactory

from .shg_pl_trans_image import SHGPLTransImage
from .shg_pl_trans_analyser import SHGPLTransAnalyser
from .shg_pl_trans_reader import SHGPLTransReader
from .shg_pl_trans_parser import SHGPLTransParser
from .shg_pl_trans_viewer import SHGPLTransViewer


class SHGPLTransFactory(BaseMultiImageFactory):

    def get_label(self):
        return 'SHG-PL-Trans'

    def get_multi_image(self):
        """Returns BaseMultiImage class for this factory"""
        return SHGPLTransImage

    def get_parser(self):
        """Returns BaseParser class able to collate images"""
        return SHGPLTransParser

    def get_reader(self):
        """Returns BaseMultiImageReader classes able to load
        the BaseMultiImage class created by this factory"""
        return SHGPLTransReader

    def get_analyser(self):
        """Returns BaseMultiImageAnalyser classes able to analyse
        the BaseMultiImage class created by this factory"""
        return SHGPLTransAnalyser

    def get_viewer(self):
        """Returns BaseMultiImageViewer class able to display
        the BaseMultiImage class created by this factory"""
        return SHGPLTransViewer
