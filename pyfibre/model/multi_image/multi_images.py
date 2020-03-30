from traits.api import (
    Array, Property, Dict, Str, ArrayOrNone
)

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.tools.preprocessing import clip_intensities
from pyfibre.model.tools.segmentation import (
    shg_segmentation, shg_pl_segmentation,
    shg_pl_trans_segmentation)


class SHGImage(BaseMultiImage):

    shg_image = Property(Array, depends_on='image_stack')

    image_dict = Property(
        Dict(Str, ArrayOrNone),
        depends_on='image_stack')

    _max_len = 1

    def _image_stack_default(self):
        return [None] * self._max_len

    def _segmentation_algorithm_default(self):
        return shg_segmentation

    def _get_shg_image(self):
        return self.image_stack[0]

    def _set_shg_image(self, image):
        self.image_stack[0] = image

    def _get_image_dict(self):
        return {
            'SHG': self.shg_image
        }

    def preprocess_images(self):
        for i, image in enumerate(self.image_stack):
            self.image_stack[i] = clip_intensities(
                image, p_intensity=self.p_intensity)


class SHGPLImage(SHGImage):

    pl_image = Property(Array, depends_on='image_stack')

    _max_len = 2

    def _segmentation_algorithm_default(self):
        return shg_pl_segmentation

    def _get_pl_image(self):
        return self.image_stack[1]

    def _set_pl_image(self, image):
        self.image_stack[1] = image

    def _get_image_dict(self):
        return {
            'SHG': self.shg_image,
            'PL': self.pl_image
        }


class SHGPLTransImage(SHGPLImage):

    trans_image = Property(Array, depends_on='image_stack')

    _max_len = 3

    def _segmentation_algorithm_default(self):
        return shg_pl_trans_segmentation

    def _get_trans_image(self):
        return self.image_stack[2]

    def _set_trans_image(self, image):
        self.image_stack[2] = image

    def _get_image_dict(self):
        return {
            'SHG': self.shg_image,
            'PL': self.pl_image,
            'Trans': self.trans_image
        }
