from pyfibre.core.base_multi_image_factory import BaseMultiImageFactory

from .multi_images import ProbeMultiImage
from .analyser import ProbeAnalyser
from .readers import ProbeMultiImageReader
from .parsers import ProbeParser
from .viewers import ProbeMultiImageViewer


class ProbeMultiImageFactory(BaseMultiImageFactory):

    def get_label(self):
        return 'Probe'

    def get_multi_image(self):
        return ProbeMultiImage

    def get_reader(self):
        return ProbeMultiImageReader

    def get_analyser(self):
        return ProbeAnalyser

    def get_parser(self):
        return ProbeParser

    def get_viewer(self):
        return ProbeMultiImageViewer
