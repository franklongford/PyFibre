import numpy as np
from skimage.io import imread

from pyfibre.shg_pl_trans.shg_image import SHGImage
from pyfibre.shg_pl_trans.shg_pl_trans_image import SHGPLTransImage
from pyfibre.shg_pl_trans.tests.fixtures import (
    test_shg_pl_trans_image_path,
    test_shg_image_path
)


class ProbeSHGImage(SHGImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        image = imread(test_shg_image_path)

        image = np.mean(image, axis=-1)
        image = image / image.max()
        image_stack = [image]

        kwargs['name'] = 'test-shg'

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

        kwargs['name'] = 'test-shg-pl-trans'

        super(ProbeSHGPLTransImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )
