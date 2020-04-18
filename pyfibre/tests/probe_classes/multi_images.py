import numpy as np
from skimage.io import imread

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.multi_image.fixed_stack_image import FixedStackImage
from pyfibre.model.multi_image.multi_images import SHGImage, SHGPLTransImage
from pyfibre.tests.fixtures import test_shg_image_path, test_shg_pl_trans_image_path

from .utilities import generate_image


class ProbeMultiImage(BaseMultiImage):

    def __init__(self, *args, **kwargs):
        image, _, _, _ = generate_image()
        kwargs['image_stack'] = [image, 2 * image]
        super().__init__(*args, **kwargs)
        self.image_dict = {
            'Test 1': self.image_stack[0],
            'Test 2': self.image_stack[1]}

    def preprocess_images(self):
        pass

    @classmethod
    def verify_stack(cls, image_stack):
        pass

    def segmentation_algorithm(self, *args, **kwargs):
        pass

    def create_figures(self, *args, **kwargs):
        pass


class ProbeSHGImage(SHGImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        image = imread(test_shg_image_path)

        image = np.mean(image, axis=-1)
        image = image / image.max()
        image_stack = [image]

        super(ProbeSHGImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )


class ProbeSHGPLTransImage(SHGPLTransImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        images = imread(test_shg_pl_trans_image_path)

        image_stack = []
        for image in images:
            image = np.mean(image, axis=-1)
            image = image / image.max()
            image_stack.append(image)

        super(ProbeSHGPLTransImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )


class ProbeFixedStackImage(FixedStackImage):

    _stack_len = 1

    _allowed_dim = [2]

    def segmentation_algorithm(self, *args, **kwargs):
        pass

    def create_figures(self, *args, **kwargs):
        pass
