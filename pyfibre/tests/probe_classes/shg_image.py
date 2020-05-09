import numpy as np
from skimage.io import imread

from pyfibre.model.multi_image.shg_image import SHGImage
from pyfibre.tests.fixtures import test_shg_image_path


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
