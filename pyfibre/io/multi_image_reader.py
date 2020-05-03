import logging
import json

import numpy as np
from skimage.util import img_as_float
from skimage.external.tifffile import TiffFile

from traits.api import HasTraits, Type

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage

logger = logging.getLogger(__name__)


class WrongFileTypeError(Exception):
    pass


def lookup_page(tiff_page):
    """Obtain relevant information from a TiffPage object"""

    xy_dim = tiff_page.shape
    description = tiff_page.image_description.decode('utf-8')

    return xy_dim, description


def get_tiff_param(tiff_file):
    """Obtain relevant parameters of TiffFile object"""

    xy_dim, description = lookup_page(tiff_file.pages[0])

    if tiff_file.is_fluoview:
        desc_list = description.split('\n')
        channel_lines = [
            line.strip() for line in desc_list if 'Gamma' in line]
        n_modes = len(channel_lines)
    else:
        # We are using test data
        desc_dict = json.loads(description)

        minor_axis = desc_dict['minor_axis']
        n_modes = desc_dict['n_modes']
        xy_dim = tuple(desc_dict['xy_dim'])

        return minor_axis, n_modes, xy_dim

    image = tiff_file.asarray()

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

    # If there is an exact match in the image shape, identify
    # this as the axis containing each mode
    elif image.shape.count(n_modes) == 1:
        major_axis = image.shape.index(n_modes)

    # If multiple image dimensions share the same number of
    # elements as number of modes, identify which corresponds
    # to each mode
    else:
        if image.shape[0] == n_modes:
            major_axis = 0
        else:
            raise IndexError(
                f"Image shape {image.shape} not supported")

    # Work out the minor axis (stack to average over) from the
    # remaining image dimensions
    minor_axes = [
        index for index, value in enumerate(image.shape)
        if value not in xy_dim and index != major_axis]

    if len(minor_axes) == 0:
        minor_axis = None
    elif len(minor_axes) == 1:
        minor_axis = minor_axes[0]
    else:
        raise IndexError(
            f"Image shape {image.shape} not supported")

    return minor_axis, n_modes, xy_dim


class MultiImageReader(HasTraits):
    """File reader that loads a stack of Tiff images, represented
    by a BaseMultiImage subclass"""

    _multi_image_class = Type(BaseMultiImage)

    def _load_images(self, filenames):
        """Load each TIFF image in turn and perform
        averaging over each stack component if required"""
        if isinstance(filenames, str):
            filenames = [filenames]

        images = []
        for filename in filenames:

            logger.info(f'Loading {filename}')

            if not self.can_load(filename):
                raise WrongFileTypeError

            with TiffFile(filename) as tiff_file:
                image = tiff_file.asarray()
                minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)

            # Add file image to stack
            images.append(self._format_image(image, minor_axis))

            logger.debug(f"Number of image modes = {n_modes}")
            logger.debug(f"Size of image = {xy_dim}")
            if minor_axis is not None:
                n_stacks = image.shape[minor_axis]
                logger.debug(f"Number of stacks = {n_stacks}")

        return images

    def _format_image(self, image, minor_axis=None):
        """Transform image to normalised float array and average
        over any stack"""

        # Average over minor axis if needed
        if minor_axis is not None:
            image = np.mean(image, axis=minor_axis)

        # If 2D array, simply normalise and return as float
        if image.ndim == 2:
            image = image / image.max()
        elif image.ndim > 2:
            for index, channel in enumerate(image):
                image[index] = channel / channel.max()

        return img_as_float(image)

    def create_image_stack(self, filenames):
        raise NotImplementedError()

    def load_multi_image(self, filenames):
        """Image loader for MultiImage classes"""

        image_stack = self.create_image_stack(filenames)

        if not self._multi_image_class.verify_stack(image_stack):
            raise ImportError(
                f"Image stack not suitable "
                f"for type {self._multi_image_class}"
            )

        multi_image = self._multi_image_class(image_stack=image_stack)

        multi_image.preprocess_images()

        return multi_image

    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""

        try:
            with TiffFile(filename) as tiff_file:
                # Check is this is Olympus FluoView formatted
                if tiff_file.is_fluoview:
                    return True

                # Check if this is test data
                _, description = lookup_page(tiff_file.pages[0])
                desc_dict = json.loads(description)
                for key in ['minor_axis', 'n_modes', 'xy_dim']:
                    _ = desc_dict[key]
        except Exception:
            logger.info(
                'File type {} not supported')
            return False

        return True
