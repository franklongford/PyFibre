import logging

import numpy as np
from skimage.util import img_as_float
from skimage.external.tifffile import imread

from traits.api import HasTraits, Type

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage

logger = logging.getLogger(__name__)


def get_image_data(image):
    """Return the number of different modes, xy dimensions
    and index of image that contains stacks of repeats"""

    minor_axis = None

    if image.ndim == 2:
        n_modes = 1
        xy_dim = image.shape

    elif image.ndim == 3:
        if np.argmin(image.shape) == 0:
            n_modes = image.shape[0]
            xy_dim = image.shape[1:]
        else:
            n_modes = 1
            xy_dim = image.shape[:2]
            minor_axis = 2

    elif image.ndim == 4:
        if image.shape[-1] == 3:
            n_modes = image.shape[0]
            xy_dim = image.shape[1:3]
        else:
            n_modes = image.shape[0]
            xy_dim = image.shape[2:]
            minor_axis = 1

    else:
        raise IndexError(
            f"Image shape {image.shape} not supported")

    logger.info("Number of image modes = {}".format(n_modes))
    logger.debug("Size of image = {}".format(xy_dim))
    if minor_axis is not None:
        n_stacks = image.shape[minor_axis]
        logger.debug("Number of stacks = {}".format(n_stacks))

    return minor_axis, n_modes, xy_dim


class MultiImageReader(HasTraits):

    _multi_image_class = Type(BaseMultiImage)

    def _load_images(self, filenames):
        """Load each TIFF image in turn and perform
        averaging over each stack component if required"""
        if isinstance(filenames, str):
            filenames = [filenames]

        images = []
        for filename in filenames:
            image = imread(filename)
            minor_axis, _, _ = get_image_data(image)
            if 'Stack' not in filename:
                minor_axis = None
            images.append(self._format_image(image, minor_axis))

        return images

    def _format_image(self, image, minor_axis=None):
        """Transform image to normalised float array and average
        over any stack"""

        if image.ndim == 2:
            return img_as_float(image / image.max())

        elif minor_axis is not None:
            image = np.mean(image, axis=minor_axis)

        image = np.apply_along_axis(
            lambda image: image / image.max(), 0, image)

        return img_as_float(image)

    def create_image_stack(self, filenames):
        raise NotImplementedError()

    def load_multi_image(self, filenames):
        """
        Image loader for MultiImage classes
        """

        image_stack = self.create_image_stack(filenames)

        if not self._multi_image_class.verify_stack(image_stack):
            raise ImportError(
                f"Image stack not suitable "
                f"for type {self._multi_image_class}"
            )

        multi_image = self._multi_image_class(image_stack=image_stack)

        multi_image.preprocess_images()

        return multi_image
