import numpy as np

from traits.api import (
    HasTraits, ArrayOrNone, Property, Tuple, Instance,
    Either, Int, Unicode, List, Array, on_trait_change
)

from pyfibre.model.tools.preprocessing import clip_intensities


class MultiImage(HasTraits):

    image_stack = List(ArrayOrNone)

    p_intensity = Tuple((1, 99))

    shape = Property(Tuple, depends_on='image_stack')

    size = Property(Tuple, depends_on='image_stack')

    def __len__(self):
        return len(self.image_stack)

    def _get_shape(self):
        if self.image_stack:
            return self.image_stack[0].shape

    def _get_size(self):
        if self.image_stack:
            return self.image_stack[0].size

    def append(self, image):
        """Appends an image to the image_stack. If image_stack
        already contains existing images, make sure that the
        shape on the incoming image matches"""
        if self.image_stack:
            if image.shape != self.shape:
                raise ValueError(
                    f'Image shape {image.shape} is not the same as '
                    f'MultiImage shape {self.shape}')

        self.image_stack.append(image)

    def remove(self, image):
        """Removes an image to the image_stack"""
        self.image_stack.remove(image)

    def preprocess_images(self):
        for i, image in enumerate(self.image_stack):
            self.image_stack[i] = clip_intensities(
                image, p_intensity=self.p_intensity
            )


class SHGPLImage(MultiImage):

    shg_image = Property(ArrayOrNone, depends_on='image_stack')

    pl_image = Property(ArrayOrNone, depends_on='image_stack')

    def _image_stack_default(self):
        return [None, None]

    def _get_shg_image(self):
        return self.image_stack[0]

    def _get_pl_image(self):
        return self.image_stack[1]

    def assign_shg_image(self, image):
        self.image_stack[0] = image

    def assign_pl_image(self, image):
        self.image_stack[1] = image


class SHGPLTransImage(SHGPLImage):

    trans_image = Property(ArrayOrNone, depends_on='image_stack')

    def _image_stack_default(self):
        return [None, None, None]

    def _get_trans_image(self):
        return self.image_stack[2]

    def assign_trans_image(self, image):
        self.image_stack[2] = image
