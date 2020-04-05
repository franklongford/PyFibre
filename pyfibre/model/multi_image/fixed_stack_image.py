from traits.api import Tuple, Property, Dict, Str, ArrayOrNone

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.tools.preprocessing import clip_intensities


class FixedStackImage(BaseMultiImage):
    """A BaseMultiImage implementation with a fixed stack size
    and defined preprocessing algorithm"""

    p_intensity = Tuple((1, 99))

    image_dict = Property(
        Dict(Str, ArrayOrNone),
        depends_on='image_stack')

    _max_len = 0

    def _image_stack_default(self):
        return [None] * self._max_len

    def _get_image_dict(self):
        return {}

    def preprocess_images(self):
        for i, image in enumerate(self.image_stack):
            self.image_stack[i] = clip_intensities(
                image, p_intensity=self.p_intensity)
