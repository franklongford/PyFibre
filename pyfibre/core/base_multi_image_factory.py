from abc import abstractmethod

from traits.api import ABCHasStrictTraits, Type, Str, provides

from pyfibre.core.base_multi_image_reader import BaseMultiImageReader

from .base_multi_image_analyser import BaseMultiImageAnalyser
from .i_multi_image_factory import IMultiImageFactory


@provides(IMultiImageFactory)
class BaseMultiImageFactory(ABCHasStrictTraits):

    tag = Str

    reader_class = Type(BaseMultiImageReader)

    analyser_class = Type(BaseMultiImageAnalyser)

    def __init__(self, **traits):

        tag = self.get_tag()
        reader = self.get_reader()
        analyser = self.get_analyser()

        super(BaseMultiImageFactory, self).__init__(
            tag=tag,
            reader_class=reader,
            analyser_class=analyser,
            **traits
        )

    @abstractmethod
    def get_tag(self):
        """Returns key associated with this factory"""

    @abstractmethod
    def get_reader(self):
        """Returns list of BaseMultiImageReader classes able to load
        the BaseMultiImage class created by this factory"""

    @abstractmethod
    def get_analyser(self):
        """Returns list of BaseMultiImageAnalyser classes able to analyse
        the BaseMultiImage class created by this factory"""

    def create_reader(self, **kwargs):
        return self.reader_class(**kwargs)

    def create_analyser(self, **kwargs):
        return self.analyser_class(**kwargs)
