import logging

from traits.api import Array, Property

from pyfibre.shg_pl_trans_plugin.shg_image import SHGImage
from pyfibre.shg_pl_trans_plugin.tools.figures import create_shg_pl_trans_figures

from .tools.segmentation import shg_pl_trans_segmentation

logger = logging.getLogger(__name__)


class SHGPLTransImage(SHGImage):
    """Object containing information from a multi channel image
    containing SHG, PL and Transmission information"""

    #: Reference to PL image
    pl_image = Property(Array, depends_on='image_stack')

    #: Reference to Transmission image
    trans_image = Property(Array, depends_on='image_stack')

    _stack_len = 3

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

    def _create_figures_default(self):
        return create_shg_pl_trans_figures

    def segmentation_algorithm(self, *args, **kwargs):
        return shg_pl_trans_segmentation(self, *args, **kwargs)
