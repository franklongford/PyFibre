from traits.api import Interface, Type, Str

from .base_multi_image_analyser import BaseMultiImageAnalyser
from pyfibre.core.base_multi_image_reader import BaseMultiImageReader


class IMultiImageFactory(Interface):

    label = Str

    reader_class = Type(BaseMultiImageReader)

    analyser_class = Type(BaseMultiImageAnalyser)

    def get_label(self):
        """Returns key associated with this factory"""

    def get_reader(self):
        """Returns list of BaseMultiImageReader classes able to load
        the BaseMultiImage class created by this factory"""

    def get_analyser(self):
        """Returns list of BaseMultiImageAnalyser classes able to analyse
        the BaseMultiImage class created by this factory"""
