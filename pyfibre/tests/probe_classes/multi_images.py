import os

from pyfibre.core.base_multi_image import BaseMultiImage
from pyfibre.model.multi_image.fixed_stack_image import FixedStackImage

from .utilities import generate_image


class ProbeMultiImage(BaseMultiImage):

    def __init__(self, *args, **kwargs):
        if 'image_stack' not in kwargs:
            image, _, _, _ = generate_image()
            kwargs['image_stack'] = [image, 2 * image]
        kwargs['name'] = 'probe_multi_image'
        kwargs['path'] = os.path.join('path', 'to', 'analysis')

        super().__init__(*args, **kwargs)

        self.image_dict = {
            f'Test {index}': image
            for index, image in enumerate(self.image_stack)
        }

    def preprocess_images(self):
        return self.image_stack

    @classmethod
    def verify_stack(cls, image_stack):
        shapes = [image.shape == (10, 10)
                  for image in image_stack]
        return all(shapes)


class ProbeFixedStackImage(FixedStackImage):

    _stack_len = 1

    _allowed_dim = [2]
