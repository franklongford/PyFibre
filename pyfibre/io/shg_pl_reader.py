import copy
import json
import logging
import os

import numpy as np
from skimage.util import img_as_float
from skimage.external.tifffile import TiffFile

from pyfibre.io.base_multi_image_reader import (
    BaseMultiImageReader, lookup_page)
from pyfibre.model.multi_image.multi_images import (
    SHGImage, SHGPLTransImage)

logger = logging.getLogger(__name__)


def get_image_type(image_path):
    """Get type of image (PL, SHG or SHG-PL-Trans) from file name"""

    image_name = os.path.basename(image_path)

    if '-pl-shg' in image_name.lower():
        image_type = 'SHG-PL-Trans'
    elif '-pl' in image_name.lower():
        image_type = 'PL-Trans'
    elif '-shg' in image_name.lower():
        image_type = 'SHG'
    else:
        image_type = 'Unknown'

    return image_type


def extract_prefix(image_name, label):
    """Extract the prefix of image_name, before label"""
    directory = os.path.dirname(image_name)
    filename = os.path.basename(image_name)
    filename_copy = filename.lower()

    index = filename_copy.index(label.lower())
    prefix = os.path.join(directory, filename[: index])

    return prefix


def get_files_prefixes(input_files, label):
    """Get the file path and file prefix of all files
    containing label"""
    files = [filename for filename in input_files
             if label in os.path.basename(filename).lower()]
    prefixes = [extract_prefix(filename, label) for filename in files]

    return files, prefixes


def filter_input_files(input_files):

    removed_files = []

    for filename in input_files:
        if not filename.endswith('.tif'):
            removed_files.append(filename)
        elif filename.find('display') != -1:
            removed_files.append(filename)
        elif filename.find('virada') != -1:
            removed_files.append(filename)
        elif filename.find('asterisco') != -1:
            removed_files.append(filename)

    for filename in removed_files:
        input_files.remove(filename)

    return input_files


def populate_image_dictionary(input_files, image_dictionary, label, tag):
    """Populate image_dictionary argument using prefixes and filenames
    of input_files list"""

    files, prefixes = get_files_prefixes(input_files, f"-{tag.lower()}")

    for filename, prefix in zip(files, prefixes):

        if prefix not in image_dictionary:
            image_dictionary[prefix] = {}

        image_dictionary[prefix][label] = filename
        input_files.remove(filename)


def collate_image_dictionary(input_files):
    """"Automatically find all combined PL-SHG files or match
    up individual images if seperate"""

    input_files = filter_input_files(copy.copy(input_files))

    image_dictionary = {}

    populate_image_dictionary(
        input_files, image_dictionary, 'SHG-PL-Trans', 'pl-shg')

    populate_image_dictionary(
        input_files, image_dictionary, 'SHG', 'shg')

    populate_image_dictionary(
        input_files, image_dictionary, 'PL-Trans', 'pl')

    return image_dictionary


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


class SHGReader(BaseMultiImageReader):
    """Reader class for a combined SHG file"""

    _multi_image_class = SHGImage

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

    def create_image_stack(self, filenames):
        """Return a list of numpy arrays suitable for a
        SHGImage"""
        image_stack = self._load_images(filenames)

        return image_stack


class SHGPLTransReader(SHGReader):
    """Reader class for a combined PL/Transmission
    file"""

    _multi_image_class = SHGPLTransImage

    def create_image_stack(self, filenames):

        images = self._load_images(filenames)

        if len(images) == 1:
            image_stack = [
                images[0][0], images[0][1], images[0][2]]
        else:
            image_stack = [
                images[0], images[1][0], images[1][1]]

        return image_stack
