import numpy as np

from pyfibre.core.base_multi_image_reader import BaseMultiImageReader

from .multi_images import ProbeFixedStackImage


class ProbeMultiImageReader(BaseMultiImageReader):

    def get_multi_image_class(self):
        return ProbeFixedStackImage

    def collate_files(self, filenames):
        return {'probe-file': [filename]
                for filename in filenames
                if '.tif' in filename}

    def can_load(self, filename):
        return filename != 'WRONG'

    def load_image(self, filename):
        return np.ones((100, 100))
