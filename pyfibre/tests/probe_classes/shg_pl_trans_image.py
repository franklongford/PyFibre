import numpy as np
from skimage.io import imread

from pyfibre.model.multi_image.shg_pl_trans_image import SHGPLTransImage
from pyfibre.tests.fixtures import test_shg_pl_trans_image_path


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
