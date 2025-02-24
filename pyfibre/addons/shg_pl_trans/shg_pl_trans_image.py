import copy
import logging

import numpy as np
from skimage import filters
from traits.api import Array, Bool, Property

from pyfibre.model.tools.preprocessing import clip_intensities

from .shg_image import SHGImage
from .tools.figures import create_shg_pl_trans_figures

from .tools.segmentation import shg_pl_trans_segmentation

logger = logging.getLogger(__name__)


class SHGPLTransImage(SHGImage):
    """Object containing information from a multi channel image
    containing SHG, PL and Transmission information"""

    #: Reference to PL image
    pl_image = Property(Array, depends_on='image_stack')

    #: Reference to Transmission image
    trans_image = Property(Array, depends_on='image_stack')

    # FIXME: set to False by default once testing is complete
    subtract_pl = Bool(True)

    _stack_len = 3

    def _get_shg_image(self):
        """Apply PL subtraction of SHG intensities if required"""
        if self.subtract_pl:
            logger.debug("Applying PL subtraction from SHG channel")
            # Attempt to subtract PL signal from SHG by reducing intensity
            # in pixels proportional to inverse PL strength
            filtered = np.where(
                self.image_stack[1] > 0,
                (self.image_stack[0] / self.image_stack[1]),
                self.image_stack[0]
            )
            return filters.median(filtered)
        return self.image_stack[0]

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

    def preprocess_images(self):
        """Clip high and low percentile image intensities
        for each image in stack."""
        return [
            clip_intensities(
                image, p_intensity=self.p_intensity)
            for image in [
                self.shg_image,
                self.pl_image,
                self.trans_image
            ]
        ]
