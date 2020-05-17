import logging

from traits.api import (
    Array, Property, Callable
)

from pyfibre.model.multi_image.fixed_stack_image import (
    FixedStackImage)
from pyfibre.shg_pl_trans_plugin.tools.figures import create_shg_figures

from .tools.segmentation import shg_segmentation

logger = logging.getLogger(__name__)


class SHGImage(FixedStackImage):
    """Object containing information from a single channel image
    containing SHG information"""

    #: Reference to SHG image
    shg_image = Property(Array, depends_on='image_stack')

    _stack_len = 1

    _allowed_dim = [2, 3]

    create_figures = Callable()

    def _get_shg_image(self):
        return self.image_stack[0]

    def _set_shg_image(self, image):
        self.image_stack[0] = image

    def _get_image_dict(self):
        return {
            'SHG': self.shg_image
        }

    def _create_figures_default(self):
        return create_shg_figures

    def segmentation_algorithm(self, *args, **kwargs):
        return shg_segmentation(self, *args, **kwargs)
