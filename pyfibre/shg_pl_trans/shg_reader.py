import copy
import json
import logging

import numpy as np
from skimage.util import img_as_float
from skimage.external.tifffile import TiffFile

from pyfibre.core.base_multi_image_reader import (
    BaseMultiImageReader)

from .shg_image import SHGImage
from .utils import filter_input_files, create_image_dictionary

logger = logging.getLogger(__name__)


def lookup_page(tiff_page):
    """Obtain relevant information from a TiffPage object"""

    xy_dim = (tiff_page.image_width, tiff_page.image_length)
    description = tiff_page.image_description.decode('utf-8')

    return xy_dim, description


def get_fluoview_param(description, xy_dim, shape):

    desc_list = description.split('\n')
    channel_lines = [
        line.strip() for line in desc_list if 'Gamma' in line]
    n_modes = len(channel_lines)

    # If number of modes is not in image shape (typically
    # because the image only contains one mode)
    if shape.count(n_modes) == 0:
        if n_modes == 1 and len(shape) == 2:
            minor_axis = None
        elif n_modes == 1 and len(shape) == 3:
            minor_axis = np.argmin(shape)
        else:
            raise IndexError(
                f"Image shape {shape} not supported")

        return minor_axis, n_modes, xy_dim

    # If there is an exact match in the image shape, identify
    # this as the axis containing each mode
    if shape.count(n_modes) == 1:
        major_axis = shape.index(n_modes)

    # If multiple image dimensions share the same number of
    # elements as number of modes, identify which corresponds
    # to each mode
    else:
        if shape[0] == n_modes:
            major_axis = 0
        else:
            raise IndexError(
                f"Image shape {shape} not supported")

    # Work out the minor axis (stack to average over) from the
    # remaining image dimensions
    minor_axes = [
        index for index, value in enumerate(shape)
        if value not in xy_dim and index != major_axis]

    if len(minor_axes) == 0:
        minor_axis = None
    elif len(minor_axes) == 1:
        minor_axis = minor_axes[0]
    else:
        raise IndexError(
            f"Image shape {shape} not supported")

    return minor_axis, n_modes, xy_dim


def get_imagej_param(description, xy_dim, shape):

    desc_list = description.split('\n')
    slices = [
        line.strip() for line in desc_list if 'slices' in line]

    if not slices:
        raise IndexError(
            f"Image shape {shape} not supported")
    else:
        n_slices = int(slices[0].split('=')[-1])
        minor_axis = shape.index(n_slices)

    # Work out the number of modes from the
    # remaining image dimensions
    n_modes = [
        index for index, value in enumerate(shape)
        if value not in xy_dim and index != minor_axis
    ]

    if len(n_modes) == 0:
        n_modes = 1
    elif len(n_modes) == 1:
        n_modes = shape[n_modes[0]]
    else:
        raise IndexError(
            f"Image shape {shape} not supported")

    return minor_axis, n_modes, xy_dim


def get_tiff_param(tiff_file):
    """Obtain relevant parameters of TiffFile object"""

    xy_dim, description = lookup_page(tiff_file.pages[0])
    shape = tiff_file.asarray().shape

    if tiff_file.is_fluoview:
        return get_fluoview_param(description, xy_dim, shape)

    elif tiff_file.is_imagej:
        return get_imagej_param(description, xy_dim, shape)

    else:
        # We are using test data
        desc_dict = json.loads(description)

        minor_axis = desc_dict['minor_axis']
        n_modes = desc_dict['n_modes']
        xy_dim = tuple(desc_dict['xy_dim'])

        return minor_axis, n_modes, xy_dim


class SHGReader(BaseMultiImageReader):
    """Reader class for a combined SHG file"""

    def get_multi_image_class(self):
        return SHGImage

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

    def collate_files(self, input_files):

        input_files = filter_input_files(copy.copy(input_files))

        image_dictionary = create_image_dictionary(
            input_files, 'shg')

        return image_dictionary

    def load_image(self, filename):

        with TiffFile(filename) as tiff_file:
            image = tiff_file.asarray()
            minor_axis, n_modes, xy_dim = get_tiff_param(tiff_file)

        logger.debug(f"Number of image modes = {n_modes}")
        logger.debug(f"Size of image = {xy_dim}")
        if minor_axis is not None:
            n_stacks = image.shape[minor_axis]
            logger.debug(f"Number of stacks = {n_stacks}")

        image = self._format_image(image, minor_axis)

        return image

    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""

        try:
            with TiffFile(filename) as tiff_file:
                # Check is this is Olympus FluoView formatted
                if tiff_file.is_fluoview:
                    return True

                # Check is this is ImageJ formatted
                if tiff_file.is_imagej:
                    return True

                # Check if this is test data
                _, description = lookup_page(tiff_file.pages[0])
                desc_dict = json.loads(description)
                for key in ['minor_axis', 'n_modes', 'xy_dim']:
                    _ = desc_dict[key]
        except Exception as e:
            logger.info(
                f'File type not supported: {e}')
            return False

        return True
