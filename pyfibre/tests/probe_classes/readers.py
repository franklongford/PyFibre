import numpy as np

from pyfibre.core.base_multi_image_reader import BaseMultiImageReader

from .multi_images import ProbeFixedStackImage
from .parsers import ProbeFileSet


class ProbeMultiImageReader(BaseMultiImageReader):

    def get_multi_image_class(self):
        return ProbeFixedStackImage

    def get_supported_file_sets(self):
        return [ProbeFileSet]

    def get_filenames(self, file_set):
        yield file_set.registry['Probe']

    def can_load(self, filename):
        return filename != 'WRONG'

    def load_image(self, filename):
        return np.ones((100, 100))
