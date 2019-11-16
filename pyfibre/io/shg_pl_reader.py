import logging
import os
import numpy as np
import copy

from skimage.io import imread
from skimage import img_as_float

from traits.api import HasTraits, Unicode, Instance

from pyfibre.model.tools.preprocessing import clip_intensities
from pyfibre.io.multi_image import MultiLayerImage, SHGPLTransImage

logger = logging.getLogger(__name__)


def load_image(image_path):
    """Load in image as a numpy float array"""
    image = img_as_float(imread(image_path))

    return image


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


def get_files_prefixes(file_list, label):
    """Get the file path and file prefix of all files
    containing label"""
    files = [filename for filename in file_list
             if label in os.path.basename(filename).lower()]
    prefixes = [extract_prefix(filename, label) for filename in files]

    return files, prefixes


def extract_prefix(image_name, label):
    """Extract the prefix of image_name, before label"""
    directory = os.path.dirname(image_name)
    filename = os.path.basename(image_name)
    filename_copy = filename.lower()

    index = filename_copy.index(label.lower())
    prefix = directory + '/' + filename[: index]

    return prefix


class SHGPLReader(HasTraits):
    """Reader class for a combined SHG/PL/Transmission
    file"""

    filename = Unicode()

    image = Instance(SHGPLTransImage)

    def __init__(self):

        self._dim_list_shg = [2, 3, 4]
        self._dim_list_pl = [3, 4]

        self._n_mode_shg = 2
        self._n_mode_pl = 2
        self._n_mode_pl_shg = 3

        self.files = {}

    def _check_dimension(self, ndim, image_type):

        if image_type == 'SHG':
            dim_list = self._dim_list_shg
        else:
            dim_list = self._dim_list_pl

        if ndim not in dim_list:
            return False

        return True

    def _check_shape(self, shape, image_type):

        if image_type == 'PL-SHG':
            if len(shape) == 4:
                n_mode = shape[0]
            else:
                n_mode = min(shape)
            if n_mode != self._n_mode_pl_shg:
                return False

        elif image_type == 'PL':
            if len(shape) == 4:
                n_mode = shape[0]
            else:
                n_mode = min(shape)
            if n_mode != self._n_mode_pl:
                return False

        elif image_type == 'SHG':
            if len(shape) == 4:
                n_mode = shape[0]
                if n_mode != self._n_mode_shg:
                    return False
            elif len(shape) == 3:
                image_shape = shape
                n_mode = int(np.min(image_shape))
                if n_mode <= self._n_mode_shg:
                    return False

        return True

    def import_image(self, image_path, image_type):
        """
        Image importer able to automatically deal with stacks and mixed SHG/PL image types
        :param image_path:
        :return:
        """

        logger.debug(f"Loading {image_path}")
        image = load_image(image_path)
        logger.debug(f"Input image shape = {image.shape}")

        image_check = self._check_dimension(image.ndim, image_type)
        image_check *= self._check_shape(image.shape, image_type)

        if not image_check:
            raise ImportError(
                f"Image shape ({image.shape}) not suitable for type {image_type}"
            )

        if 'PL' in image_type:

            major_axis = 0

            if image.ndim == 4:
                image_shape = image[major_axis].shape
            else:
                image_shape = image.shape

            logger.info("Number of image types = {}".format(image.shape[major_axis]))

            minor_axis = int(np.argmin(image_shape))
            xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
            n_stacks = image_shape[minor_axis]

            logger.debug("Size of image = {}".format(xy_dim))
            logger.debug("Number of stacks = {}".format(n_stacks))

            if image_type == 'PL':
                if image.ndim == 4:
                    image_pl = np.mean(image[0], axis=minor_axis)
                    image_tran = np.mean(image[1], axis=minor_axis)
                else:
                    image_pl = np.take(image, 0, minor_axis)
                    image_tran = np.take(image, 1, minor_axis)

                image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
                image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

                return image_pl, image_tran

            elif image_type == 'PL-SHG':
                if image.ndim == 4:
                    image_shg = np.mean(image[0], axis=minor_axis)
                    image_pl = np.mean(image[1], axis=minor_axis)
                    image_tran = np.mean(image[2], axis=minor_axis)
                else:
                    image_shg = np.take(image, 0, minor_axis)
                    image_pl = np.take(image, 1, minor_axis)
                    image_tran = np.take(image, 2, minor_axis)

                image_shg = clip_intensities(image_shg, p_intensity=(0, 100))
                image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
                image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

                return image_shg, image_pl, image_tran

        elif image_type == 'SHG':

            if image.ndim == 4:
                major_axis = 0
                image_shape = image[major_axis].shape

                logger.info("Number of image types = {}".format(image.shape[major_axis]))

                minor_axis = np.argmin(image_shape)
                xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
                n_stacks = image_shape[minor_axis]

                logger.debug("Number of stacks = {}".format(n_stacks))

                image_shg = np.mean(image[1], axis=minor_axis)

            elif image.ndim == 3:
                image_shape = image.shape

                minor_axis = np.argmin(image_shape)
                xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
                n_stacks = image_shape[minor_axis]

                logger.debug("Number of stacks = {}".format(n_stacks))

                image_shg = np.mean(image, axis=minor_axis)

            else:
                xy_dim = image.shape
                image_shg = image

            logger.debug("Size of image = {}".format(xy_dim))
            image_shg = clip_intensities(image_shg, p_intensity=(0, 100))

            return image_shg

        raise RuntimeError('Image file not appropriately labelled')

    def _clean_files(self):
        self.files = {}

    def get_image_lists(self, input_files):
        """"Automatically find all combined PL-SHG files or match
        up individual images if seperate"""

        input_files = copy.copy(input_files)
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

        files, prefixes = get_files_prefixes(input_files, '-pl-shg')

        for filename, prefix in zip(files, prefixes):
            self.files[prefix] = {}
            self.files[prefix]['PL-SHG'] = filename
            input_files.remove(filename)

        shg_files, shg_prefixes = get_files_prefixes(input_files, '-shg')
        pl_files, pl_prefixes = get_files_prefixes(input_files, '-pl')

        for i, prefix in enumerate(shg_prefixes):

            if prefix not in self.files.keys():

                indices = [j for j, pl_prefix in enumerate(pl_prefixes)
                           if prefix in pl_prefix]

                if indices:
                    self.files[prefix] = {}
                    self.files[prefix]['SHG'] = shg_files[i]
                    self.files[prefix]['PL'] = pl_files[indices[0]]

                else:
                    self.files[prefix] = {}
                    self.files[prefix]['SHG'] = shg_files[i]

                    self.files.pop(prefix, None)
                    logger.debug(f'Could not find PL image data for {prefix}')

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

    def update_multi_images(self):
        """Update the MultiImages with new TifReader attributes"""

        for prefix, data in self.files.items():

            multi_image = data['image']

            multi_image.ow_network = self.ow_network
            multi_image.ow_segment = self.ow_segment
            multi_image.ow_metric = self.ow_metric
            multi_image.ow_figure = self.ow_figure
            multi_image.shg_analysis = self.shg
            multi_image.pl_analysis = self.pl
            multi_image.p_intensity = self.p_intensity
