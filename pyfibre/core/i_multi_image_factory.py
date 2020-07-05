from traits.api import Interface, Type, Str

from .i_file_parser import IFileParser
from .i_multi_image import IMultiImage
from .i_multi_image_analyser import IMultiImageAnalyser
from .i_multi_image_reader import IMultiImageReader
from .i_multi_image_viewer import IMultiImageViewer


class IMultiImageFactory(Interface):

    label = Str

    multi_image_class = Type(IMultiImage)

    reader_class = Type(IMultiImageReader)

    analyser_class = Type(IMultiImageAnalyser)

    parser_class = Type(IFileParser)

    viewer_class = Type(IMultiImageViewer)

    def get_label(self):
        """Returns key associated with this factory"""

    def get_multi_image(self):
        """Returns BaseMultiImage associated with this factory"""

    def get_reader(self):
        """Returns list of IMultiImageReader classes able to load
        the IMultiImage class created by this factory"""

    def get_analyser(self):
        """Returns list of IMultiImageAnalyser classes able to analyse
        the IMultiImage class created by this factory"""

    def get_parser(self):
        """Returns list of IFileParser classes able to parse files
        that are used to load the IMultiImage class created by
        this factory"""

    def get_viewer(self):
        """Returns list of IMultiImageViewer classes able to display
        the IMultiImage class created by this factory"""
