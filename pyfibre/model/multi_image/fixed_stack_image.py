from traits.api import Tuple, Property, Dict, Str, ArrayOrNone

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.tools.preprocessing import clip_intensities


class FixedStackImage(BaseMultiImage):
    """A BaseMultiImage implementation with a fixed stack size
    and defined preprocessing algorithm"""

    #: Percentages of image intensity to be rescaled between
    #: in preprocess_images method
    p_intensity = Tuple((1, 99))

    #: Dictionary containing references to each entry in
    #: image_stack as a property
    image_dict = Property(
        Dict(Str, ArrayOrNone),
        depends_on='image_stack')

    _stack_len = 0

    _allowed_dim = []

    def _image_stack_default(self):
        return [None] * self._stack_len

    def _get_image_dict(self):
        return {}

    @classmethod
    def verify_stack(cls, image_stack):
        """Perform verification that image_stack is allowed by
        subclass of BaseMultiImage"""
        if len(image_stack) != cls._stack_len:
            return False

        for image in image_stack:
            if image.shape != image_stack[0].shape:
                return False
            if image.ndim not in cls._allowed_dim:
                return False

        return True

    def preprocess_images(self):
        for i, image in enumerate(self.image_stack):
            self.image_stack[i] = clip_intensities(
                image, p_intensity=self.p_intensity)
