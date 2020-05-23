import numpy as np

from pyfibre.io.core.base_multi_image_reader import BaseMultiImageReader

from .multi_images import ProbeFixedStackImage


class ProbeMultiImageReader(BaseMultiImageReader):

    _multi_image_class = ProbeFixedStackImage

    def can_load(self, filename):
        return True

    def load_image(self, filename):
        return np.ones((100, 100))

    def create_image_stack(self, filenames):
        return [self.load_image(filename) for filename in filenames]
