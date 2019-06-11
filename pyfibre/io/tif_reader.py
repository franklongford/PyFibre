import logging
import numpy as np

from skimage import io

logger = logging.getLogger(__name__)


class TIFReader():

    def __init__(self):
        pass

    def _get_image_type(self, image_path):

        image_name = image_path.split('/')[-1]

        if '-pl-shg' in image_name.lower():
            image_type = 'PL-SHG'
        elif '-pl' in image_name.lower():
            image_type = 'PL'
        elif '-shg' in image_name.lower():
            image_type = 'SHG'
        else:
            raise RuntimeError('Image file not appropriately labelled')

        return image_type

    def _load_image(self, image_path):
        logger.debug(f"Loading {image_path}")
        image = io.imread(image_path).astype(np.float64)
        logger.debug(f"Input image shape = {image.shape}")

        return image

    def _check_dimension(self, ndim, image_type):

        dim_list = [3, 4]
        if image_type == 'SHG':
            dim_list += [2]

        if ndim not in dim_list:
            raise ImportError(
                f"Image dimensions ({ndim}) not suitable for type {image_type}"
            )

        return True

    def _check_shape(self, shape, image_type):

        if len(shape) == 4:
            major_axis = 0
            image_shape = image[major_axis].shape
        elif len(shape) == 3:
            major_axis = int(np.argmin(image.shape))
            image_shape = image.shape

        if image.shape[axis] != n:
            raise ImportError(
                f"Image shape ({image.shape}) not suitable for type {image_type}"
            )

    def import_image(self, image_path):
        """
        Image importer able to automatically deal with stacks and mixed SHG/PL image types
        :param image_path:
        :return:
        """

        image_type = self._get_image_type(image_path)
        image = self._load_image(image_path)

        if image_type == 'PL-SHG':

            self._check_dimension(image.ndim, image_type)

            if image.ndim == 4:
                major_axis = 0
                image_shape = image[major_axis].shape
            elif image.ndim == 3:
                major_axis = int(np.argmin(image.shape))
                image_shape = image.shape

            logger.info("Number of image types = {}".format(image.shape[major_axis]))

            self._check_shape(image_shape, image_type)

            minor_axis = int(np.argmin(image_shape))
            xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
            n_stacks = image_shape[minor_axis]

            logger.debug("Size of image = {}".format(xy_dim))
            logger.debug("Number of stacks = {}".format(n_stacks))

            if image.ndim == 4:
                image_shg = np.mean(image[0], axis=minor_axis)
                image_pl = np.mean(image[1], axis=minor_axis)
                image_tran = np.mean(image[2], axis=minor_axis)

            elif image.ndim == 3:
                image_shg = np.take(image, 0, minor_axis)
                image_pl = np.take(image, 1, minor_axis)
                image_tran = np.take(image, 2, minor_axis)

            image_shg = clip_intensities(image_shg, p_intensity=(0, 100))
            image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
            image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

            return image_shg, image_pl, image_tran

        elif image_type == 'PL':

            self._check_dimension(image.ndim, image_type)

            if image.ndim == 4:
                major_axis = 0
                image_shape = image[major_axis].shape
            elif image.ndim == 3:
                image_shape = image.shape
                major_axis = int(np.argmin(image_shape))

            logger.info("Number of image types = {}".format(image.shape[major_axis]))

            self._check_shape(image, major_axis, 2, image_type)

            minor_axis = int(np.argmin(image_shape))
            xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
            n_stacks = image_shape[minor_axis]

            logger.debug("Size of image = {}".format(xy_dim))
            logger.debug("Number of stacks = {}".format(n_stacks))

            if image.ndim == 4:
                image_pl = np.mean(image[0], axis=minor_axis)
                image_tran = np.mean(image[1], axis=minor_axis)

            elif image.ndim == 3:
                image_pl = np.take(image, 0, minor_axis)
                image_tran = np.take(image, 1, minor_axis)

            image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
            image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

            return image_pl, image_tran

        elif image_type == 'SHG':

            self._check_dimension(image.ndim, image_type)

            if image.ndim == 4:
                major_axis = 0
                image_shape = image[major_axis].shape

                logger.info("Number of image types = {}".format(image.shape[minor_axis]))

                self._check_shape(image, major_axis, 2, image_type)

                minor_axis = np.argmin(image_shape)
                xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
                n_stacks = image_shape[minor_axis]

                logger.debug("Size of image = {}".format(xy_dim))
                logger.debug("Number of stacks = {}".format(n_stacks))

                image_shg = np.mean(image[1], axis=minor_axis)

            elif image.ndim == 3:
                image_shape = image.shape

                minor_axis = np.argmin(image_shape)
                xy_dim = tuple(x for i, x in enumerate(image_shape) if i != minor_axis)
                n_stacks = image_shape[minor_axis]

                logger.debug("Size of image = {}".format(xy_dim))
                logger.debug("Number of stacks = {}".format(n_stacks))

                image_shg = np.mean(image, axis=minor_axis)

            elif image.ndim == 2:
                logger.debug("Size of image = {}".format(image_orig.shape))
                image_shg = image

            image_shg = clip_intensities(image_shg, p_intensity=(0, 100))

            return image_shg

        raise IOError

    def load_multi_image(self, input_file_names):
        "Load in SHG and PL files from file name tuple"

        image_stack = [None, None, None]

        for filename in input_file_names:
            if '-pl-shg' in filename.lower():
                image_stack[0], image_stack[1], image_stack[2] = self.import_image(filename)
            elif '-shg' in filename.lower():
                image_stack[0] = self.import_image(filename)
            elif '-pl' in filename.lower():
                image_stack[1], image_stack[2] = self.import_image(filename)
            else:
                raise RuntimeError('Image file not appropriately labelled')

        return image_stack