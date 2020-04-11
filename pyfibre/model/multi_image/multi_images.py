from traits.api import (
    Array, Property
)

from pyfibre.model.multi_image.fixed_stack_image import FixedStackImage
from pyfibre.model.tools.segmentation import (
    shg_segmentation,
    shg_pl_trans_segmentation)
from pyfibre.model.tools.figures import (
    create_shg_figures, create_shg_pl_trans_figures)


class SHGImage(FixedStackImage):

    shg_image = Property(Array, depends_on='image_stack')

    _stack_len = 1

    _allowed_dim = [2, 3]

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

    def create_figures(self, figname, **kwargs):
        return create_shg_figures(self, figname, **kwargs)


class SHGPLTransImage(SHGImage):

    pl_image = Property(Array, depends_on='image_stack')

    trans_image = Property(Array, depends_on='image_stack')

    _stack_len = 3

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

    def create_figures(self, figname, **kwargs):
        return create_shg_pl_trans_figures(self, figname, **kwargs)