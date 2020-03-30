import copy
import logging
import os

import numpy as np
from skimage import img_as_float

from traits.api import (
    File, Unicode, List, Enum, Property)
from traitsui.api import View, Item

from pyfibre.io.multi_image_reader import MultiImageReader
from pyfibre.model.multi_image.multi_images import (
    SHGPLImage, SHGPLTransImage)

logger = logging.getLogger(__name__)


def get_image_type(image_path):
    """Get type of image (PL, SHG or PL-SHG) from file name"""

    image_name = os.path.basename(image_path)

    if '-pl-shg' in image_name.lower():
        image_type = 'PL-SHG'
    elif '-pl' in image_name.lower():
        image_type = 'PL'
    elif '-shg' in image_name.lower():
        image_type = 'SHG'
    else:
        image_type = 'Unknown'

    return image_type


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
        n_modes = image.shape[0]
        xy_dim = image.shape[1:3]
        minor_axis = image.shape[3]

    else:
        raise IndexError(
            f"Image shape {image.shape} not supported")

    logger.info("Number of image modes = {}".format(n_modes))
    logger.debug("Size of image = {}".format(xy_dim))
    if minor_axis is not None:
        n_stacks = image.shape[minor_axis]
        logger.debug("Number of stacks = {}".format(n_stacks))

    return minor_axis, n_modes, xy_dim


def extract_prefix(image_name, label):
    """Extract the prefix of image_name, before label"""
    directory = os.path.dirname(image_name)
    filename = os.path.basename(image_name)
    filename_copy = filename.lower()

    index = filename_copy.index(label.lower())
    prefix = directory + '/' + filename[: index]

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


def populate_image_dictionary(input_files, image_dictionary, label):
    """Populate image_dictionary argument using prefixes and filenames
    of input_files list"""

    files, prefixes = get_files_prefixes(input_files, f"-{label.lower()}")

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

    populate_image_dictionary(input_files, image_dictionary, 'PL-SHG')

    populate_image_dictionary(input_files, image_dictionary, 'SHG')

    populate_image_dictionary(input_files, image_dictionary, 'PL')

    return image_dictionary


class SHGPLReader(MultiImageReader):
    """Reader class for a combined SHG/PL/Transmission
    file"""

    load_mode = Enum('PL-SHG File', 'Separate Files')

    shg_pl_filename = File()

    shg_filename = File()

    pl_filename = File()

    filenames = Property(List(File),
                         depends_on='load_mode,shg_pl_filename,'
                                    'shg_filename,pl_filename')

    image_types = Property(List(Unicode),
                           depends_on='load_mode')

    traits_view = View(
        Item('load_mode'),
        Item('shg_pl_filename', visible_when="load_mode=='PL-SHG File'"),
        Item('shg_filename', visible_when="load_mode=='Separate Files'"),
        Item('pl_filename', visible_when="load_mode=='Separate Files'")
    )

    def _get_filenames(self):
        """Switch files to be loaded, according to loading mode"""
        if self.load_mode == 'PL-SHG File':
            return [self.shg_pl_filename]
        else:
            return [self.shg_filename, self.pl_filename]

    def _get_image_types(self):
        if self.load_mode == "PL-SHG File":
            return ['PL-SHG']
        else:
            return ['SHG', 'PL']

    def _multi_image_class_default(self):
        return SHGPLImage

    def _check_dimension(self, n_dim, image_type):
        """Assert that dimension of an image is allowed for
        selected image type"""

        if image_type == 'PL-SHG':
            return n_dim in [3, 4]
        else:
            return n_dim in [2, 3, 4]

    def _check_n_modes(self, n_modes, image_type):
        """Assert that the number of modes contained in an image
        is allowed for selected image type"""

        if image_type == 'PL-SHG':
            return n_modes in [2, 3]
        else:
            return n_modes in [1, 2]

    def _format_image(self, image, n_modes, minor_axis):

        if image.ndim == 4:
            image_stack = tuple(
                np.mean(image[index], axis=minor_axis-1)
                for index in range(n_modes))

        elif image.ndim == 3:
            if minor_axis is None:
                image_stack = tuple(
                    image[index]
                    for index in range(n_modes))
            else:
                image_stack = (np.mean(image, axis=minor_axis),)
        else:
            image_stack = (image,)

        image_stack = [img_as_float(image / image.max())
                       for image in image_stack]

        return image_stack

    def image_preprocessing(self, images):

        image_stack = []

        for image, image_type in zip(images, self.image_types):

            minor_axis, n_modes, xy_dim = get_image_data(image)

            image_check = self._check_dimension(image.ndim, image_type)
            image_check *= self._check_n_modes(n_modes, image_type)

            if not image_check:
                raise ImportError(
                    f"Image shape ({image.shape}) not suitable "
                    f"for type {image_type}"
                )

            if image_type == 'PL-SHG':
                image_stack += self._format_image(
                    image, n_modes, minor_axis)
            else:
                image_stack += self._format_image(
                    image, 1, minor_axis)

        return image_stack

    def assign_images(self, image_dictionary):
        """Assign images from an image_dictionary to
        the PL-SHG, PL and SHG file names"""

        if 'PL-SHG' in image_dictionary:
            self.shg_pl_filename = image_dictionary['PL-SHG']
        if 'PL' in image_dictionary:
            self.pl_filename = image_dictionary['PL']
        if 'SHG' in image_dictionary:
            self.shg_filename = image_dictionary['SHG']


class SHGPLTransReader(SHGPLReader):

    def _multi_image_class_default(self):
        return SHGPLTransImage

    def image_preprocessing(self, images):

        image_stack = []

        for image, image_type in zip(images, self.image_types):

            minor_axis, n_modes, xy_dim = get_image_data(image)

            image_check = self._check_dimension(image.ndim, image_type)
            image_check *= self._check_n_modes(n_modes, image_type)

            if not image_check:
                raise ImportError(
                    f"Image shape ({image.shape}) not suitable "
                    f"for type {image_type}"
                )

            if image_type == 'SHG':
                image_stack += self._format_image(
                    image, 1, minor_axis)
            else:
                image_stack += self._format_image(
                    image, n_modes, minor_axis)

        return image_stack


def load_multi_images(self):
    """Load in SHG and PL files from file name tuple"""

    remove_image = []
    remove_shg = []
    remove_pl = []

    for prefix, data in self.files.items():

        image_stack = [None, None, None]

        multi_image = SHGPLTransImage()

        if 'PL-SHG' in data:

            (image_stack[0],
             image_stack[1],
             image_stack[2]) = self.import_image(data['PL-SHG'], 'PL-SHG')

            multi_image.file_path = prefix
            multi_image.assign_shg_image(image_stack[0])
            multi_image.shg = image_stack[0]
            multi_image.image_pl = image_stack[1]
            multi_image.image_tran = image_stack[2]

        else:
            if 'SHG' in data:
                try:
                    image_stack[0] = self.import_image(data['SHG'], 'SHG')
                    multi_image.file_path = prefix
                    multi_image.image_shg = image_stack[0]
                    multi_image.preprocess_image_shg()
                except ImportError:
                    logger.debug('Unable to load SHG file')
                    remove_shg.append(prefix)
                    multi_image.shg_analysis = False
                    if self.shg:
                        remove_image.append(prefix)
            else:
                logger.debug(
                    'SHG Image file not appropriately labelled')
                multi_image.shg_analysis = False
                if self.shg:
                    remove_image.append(prefix)

            if 'PL' in data:
                try:
                    (image_stack[1],
                     image_stack[2]) = self.import_image(data['PL'], 'PL')
                    multi_image.image_pl = image_stack[1]
                    multi_image.image_tran = image_stack[2]
                    multi_image.preprocess_image_pl()
                except ImportError:
                    logger.debug('Unable to load PL file')
                    remove_pl.append(prefix)
                    multi_image.pl_analysis = False
                    if self.pl:
                        remove_image.append(prefix)
            else:
                logger.debug(
                    'PL Image file not appropriately labelled')
                multi_image.pl_analysis = False
                if self.pl:
                    remove_image.append(prefix)

        self.files[prefix]['image'] = multi_image

    [self.files[key].pop('SHG', None) for key in remove_shg]
    [self.files[key].pop('PL', None) for key in remove_pl]
    [self.files.pop(key, None) for key in remove_image]
