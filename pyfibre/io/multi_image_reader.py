import logging

import numpy as np
from skimage.util import img_as_float
from skimage.external.tifffile import TiffFile

from traits.api import HasTraits, Type

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage

logger = logging.getLogger(__name__)


def get_image_data(image):
    """Return the number of different modes, xy dimensions
    and index of image that contains stacks of repeats."""

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

    return minor_axis, n_modes, xy_dim


def lookup_page(tiff_page):
    """Obtain relevant information from a TiffPage object"""

    xy_dim = tiff_page.shape
    description = tiff_page.image_description.decode()

    return xy_dim, description


def get_tiff_param(tiff_file):
    """Obtain relevant parameters of TiffFile object"""

    xy_dim, description = lookup_page(tiff_file.pages[0])
    image = tiff_file.asarray()

    if tiff_file.is_fluoview:
        desc_list = description.split('\n')
        channel_lines = [
            line.strip() for line in desc_list if 'Gamma' in line]
        n_modes = len(channel_lines)
    else:
        # Check if this is test data
        try:
            n_modes = description['n_modes']
        except Exception:
            raise RuntimeError(
                'Only Olympus Tiff images currently supported')

    # If number of modes is not in image shape (typically
    # because the image only contains one mode)
    if image.shape.count(n_modes) == 0:
        if n_modes == 1 and image.ndim == 2:
            minor_axis = None
        elif n_modes == 1 and image.ndim == 3:
            minor_axis = np.argmin(image.shape)
        else:
            raise IndexError(
                f"Image shape {image.shape} not supported")

        return minor_axis, n_modes, xy_dim

    elif image.shape.count(n_modes) == 1:
        major_axis = image.shape.index(n_modes)

    else:
        if image.shape[0] == n_modes:
            major_axis = 0
        else:
            raise IndexError(
                f"Image shape {image.shape} not supported")

    minor_axes = [
        index for index, value in enumerate(image.shape)
        if value not in xy_dim and index != major_axis]

    if len(minor_axes) == 1:
        minor_axis = minor_axes[0]
    else:
        raise IndexError(
            f"Image shape {image.shape} not supported")

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
            logger.info(f'Loading {filename}')
            with TiffFile(filename) as tiff_file:
                image = tiff_file.asarray()
                minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)

            logger.debug(f"Number of image modes = {n_modes}")
            logger.debug(f"Size of image = {xy_dim}")
            if minor_axis is not None:
                n_stacks = image.shape[minor_axis]
                logger.debug(f"Number of stacks = {n_stacks}")

            # if 'Stack' not in filename:
            #    minor_axis = None
            images.append(self._format_image(image, minor_axis))

        return images

    def _format_image(self, image, minor_axis=None):
        """Transform image to normalised float array and average
        over any stack"""

        if image.ndim == 2:
            return img_as_float(image / image.max())

        elif minor_axis is not None:
            image = np.mean(image, axis=minor_axis)

        if image.ndim > 2:
            for index, channel in enumerate(image):
                image[index] = channel / channel.max()
        else:
            image = image / image.max()

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
