from traits.api import Interface, Type, Str

from .i_multi_image_analyser import IMultiImageAnalyser
from .i_multi_image_reader import IMultiImageReader


class IMultiImageFactory(Interface):

    label = Str

    reader_class = Type(IMultiImageReader)

    analyser_class = Type(IMultiImageAnalyser)

    def get_label(self):
        """Returns key associated with this factory"""

    def get_reader(self):
        """Returns list of IMultiImageReader classes able to load
        the IMultiImage class created by this factory"""

    def get_analyser(self):
        """Returns list of IMultiImageAnalyser classes able to analyse
        the IMultiImage class created by this factory"""
