from abc import abstractmethod

from traits.api import ABCHasStrictTraits, Type, Str, provides

from .i_file_parser import IFileParser
from .i_multi_image import IMultiImage
from .i_multi_image_analyser import IMultiImageAnalyser
from .i_multi_image_factory import IMultiImageFactory
from .i_multi_image_reader import IMultiImageReader
from .i_multi_image_viewer import IMultiImageViewer


@provides(IMultiImageFactory)
class BaseMultiImageFactory(ABCHasStrictTraits):
    """Main component contributed by plugins to allow expansion
    of the software. Represents a multi-channel image that can be
    loaded from a single or multiple files, with an analysis
    routine.
    """

    #: Label to be displayed in UI
    label = Str()

    #: Multi image class type associated with this factory
    multi_image_class = Type(IMultiImage)

    #: Reader class, used to load a BaseMultiImage from file
    reader_class = Type(IMultiImageReader)

    #: Analyser class, used to perform an analysis script on
    #: a specific image type
    analyser_class = Type(IMultiImageAnalyser)

    #: Parser class, used to collate files into sets
    parser_class = Type(IFileParser)

    #: Viewer class, used to display an image type
    viewer_class = Type(IMultiImageViewer)

    def __init__(self, **traits):

        label = self.get_label()
        multi_image = self.get_multi_image()
        reader = self.get_reader()
        analyser = self.get_analyser()
        parser = self.get_parser()
        viewer = self.get_viewer()

        super(BaseMultiImageFactory, self).__init__(
            label=label,
            multi_image_class=multi_image,
            reader_class=reader,
            analyser_class=analyser,
            parser_class=parser,
            viewer_class=viewer,
            **traits
        )

    @abstractmethod
    def get_label(self):
        """Returns key associated with this factory"""

    @abstractmethod
    def get_multi_image(self):
        """Returns BaseMultiImage associated with this factory"""

    @abstractmethod
    def get_reader(self):
        """Returns BaseMultiImageReader class able to load
        the BaseMultiImage class created by this factory"""

    @abstractmethod
    def get_analyser(self):
        """Returns BaseMultiImageAnalyser class able to analyse
        the BaseMultiImage class created by this factory"""

    @abstractmethod
    def get_parser(self):
        """Returns BaseFileParser class able to collate image files
        together"""

    @abstractmethod
    def get_viewer(self):
        """Returns BaseMultiImageViewer class able to display
        the BaseMultiImage class created by this factory"""

    def create_reader(self, **kwargs):
        """Public method used to return an instance of
        BaseMultiImageReader"""
        return self.reader_class(**kwargs)

    def create_analyser(self, **kwargs):
        """Public method used to return an instance of
        BaseMultiImageAnalyser"""
        return self.analyser_class(**kwargs)

    def create_parser(self, **kwargs):
        """Public method used to return an instance of
        BaseFileParser"""
        return self.parser_class(**kwargs)

    def create_viewer(self, **kwargs):
        """Public method used to return an instance of
        BaseMultiImageViewer"""
        return self.viewer_class(**kwargs)
