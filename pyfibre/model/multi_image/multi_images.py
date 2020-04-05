from traits.api import (
    Array, Property
)

from pyfibre.model.multi_image.fixed_stack_image import FixedStackImage
from pyfibre.model.tools.segmentation import (
    shg_segmentation,
    shg_pl_trans_segmentation)


class SHGImage(FixedStackImage):

    shg_image = Property(Array, depends_on='image_stack')

    _max_len = 1

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


class PLTransImage(FixedStackImage):

    pl_image = Property(Array, depends_on='image_stack')

    trans_image = Property(Array, depends_on='image_stack')

    _max_len = 2

    def _segmentation_algorithm_default(self):
        return None

    def _get_pl_image(self):
        return self.image_stack[0]

    def _set_pl_image(self, image):
        self.image_stack[0] = image

    def _get_trans_image(self):
        return self.image_stack[1]

    def _set_trans_image(self, image):
        self.image_stack[1] = image

    def _get_image_dict(self):
        return {
            'PL': self.pl_image,
            'Trans': self.trans_image
        }


class SHGPLTransImage(SHGImage, PLTransImage):

    _max_len = 3

    def _segmentation_algorithm_default(self):
        return shg_pl_trans_segmentation

    def _get_pl_image(self):
        return self.image_stack[1]

    def _set_pl_image(self, image):
        self.image_stack[1] = image

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
