import copy
import logging
import os

import numpy as np
from skimage import img_as_float
from traits.api import (
    File, Type, HasTraits,
    List, Enum, Property)
from traitsui.api import View, Item

from pyfibre.model.objects.multi_image import SHGPLImage
#from .multi_image_reader import MultiImageReader

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


class SHGPLReader(HasTraits):
    """Reader class for a combined SHG/PL/Transmission
    file"""

    load_mode = Enum('SHG-PL File', 'Separate Files')

    shg_pl_filename = File()

    shg_filename = File()

    pl_filename = File()

    filenames = Property(List(File), depends_on='load_mode,shg_pl_filename,'
                                                'shg_filename,pl_filename')

    multi_image_class = Type(SHGPLImage)

    traits_view = View(
        Item('load_mode'),
        Item('shg_pl_filename', visible_when="load_mode=='SHG-PL File'"),
        Item('shg_filename', visible_when="load_mode=='Separate Files'"),
        Item('pl_filename', visible_when="load_mode=='Separate Files'")
    )

    def __init__(self, *args, **kwargs):
        super(SHGPLReader, self).__init__(*args, **kwargs)

        self._n_mode_shg = 2
        self._n_mode_pl = 2
        self._n_mode_pl_shg = 3

    def _get_filenames(self):
        if self.load_mode == 'Load File':
            return [self.shg_pl_filename]
        else:
            return [self.shg_filename, self.pl_filename]

    def _check_dimension(self, ndim, image_type):

        dim_list = [2, 3, 4]
        if 'PL' in image_type:
            dim_list.pop(0)

        return ndim in dim_list

    def _check_shape(self, shape, image_type):

        if len(shape) == 4:
            n_mode = shape[0]
        elif len(shape) == 3:
            n_mode = min(shape)
        else:
            return True

        if image_type == 'PL-SHG':
            return n_mode == self._n_mode_pl_shg
        elif image_type == 'PL':
            return n_mode == self._n_mode_pl
        else:
            return n_mode == self._n_mode_shg

    def _get_minor_axis(self, image):

        if image.ndim == 2:
            xy_dim = image.shape
            minor_axis = None
        else:
            if image.ndim == 4:
                logger.info("Number of image types = {}".format(image.shape[0]))
                image_shape = image.shape[1]
            else:
                image_shape = image.shape

            minor_axis = int(np.argmin(image_shape))
            xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)

            n_stacks = image_shape[minor_axis]
            logger.debug("Number of stacks = {}".format(n_stacks))

        logger.debug("Size of image = {}".format(xy_dim))

        return minor_axis

    def _format_shg_pl_image(self, image, minor_axis):

        if image.ndim == 4:
            image_shg = np.mean(image[0], axis=minor_axis)
            image_pl = np.mean(image[1], axis=minor_axis)
            image_tran = np.mean(image[2], axis=minor_axis)
        else:
            image_shg = np.take(image, 0, minor_axis)
            image_pl = np.take(image, 1, minor_axis)
            image_tran = np.take(image, 2, minor_axis)

        return (
            img_as_float(image_shg),
            img_as_float(image_pl),
            img_as_float(image_tran))

    def _format_shg_image(self, image, minor_axis):

        if minor_axis is None:
            image_shg = image
        elif image.ndim == 4:
            image_shg = np.mean(image[1], axis=minor_axis)
        else:
            image_shg = np.mean(image, axis=minor_axis)

        return img_as_float(image_shg)

    def _format_pl_image(self, image, minor_axis):

        if image.ndim == 4:
            image_pl = np.mean(image[0], axis=minor_axis)
            image_tran = np.mean(image[1], axis=minor_axis)
        else:
            image_pl = np.take(image, 0, minor_axis)
            image_tran = np.take(image, 1, minor_axis)

        return (img_as_float(image_pl),
                img_as_float(image_tran))

    def image_preprocessing(self, image, image_type):

        image_check = self._check_dimension(image.ndim, image_type)
        image_check *= self._check_shape(image.shape, image_type)

        if not image_check:
            raise ImportError(
                f"Image shape ({image.shape}) not suitable for type {image_type}"
            )

        minor_axis = self._get_minor_axis(image)

        if image_type == 'PL-SHG':
            images = self._format_shg_pl_image(image, minor_axis)
        elif image_type == 'PL':
            images = self._format_pl_image(image, minor_axis)
        else:
            images = self._format_shg_image(image, minor_axis)

        return images

    def load_multi_image(self):
        """
        Image loader able to automatically deal with stacks and mixed SHG/PL image types
        """

        if self.load_mode == "SHG-PL File":
            image_types = ['SHG-PL']
        else:
            image_types = ['SHG', 'PL']

        images = self.load_images()

        for index, image, image_type in enumerate(images):
            image_type = image_types[index]

            logger.debug(f"Input image shape = {image.shape}")

            images[index] = self.image_preprocessing(image, image_type)

        multi_image = self.multi_image_class(image_stack=images)

        return multi_image


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
                logger.debug('SHG Image file not appropriately labelled')
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
                logger.debug('PL Image file not appropriately labelled')
                multi_image.pl_analysis = False
                if self.pl:
                    remove_image.append(prefix)

        self.files[prefix]['image'] = multi_image

    [self.files[key].pop('SHG', None) for key in remove_shg]
    [self.files[key].pop('PL', None) for key in remove_pl]
    [self.files.pop(key, None) for key in remove_image]
