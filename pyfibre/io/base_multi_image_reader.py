from abc import abstractmethod
import logging

from traits.api import ABCHasTraits, Type

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage

logger = logging.getLogger(__name__)


class WrongFileTypeError(Exception):
    pass


def lookup_page(tiff_page):
    """Obtain relevant information from a TiffPage object"""

    xy_dim = tiff_page.shape
    description = tiff_page.image_description.decode('utf-8')

    return xy_dim, description


class BaseMultiImageReader(ABCHasTraits):
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

            image = self.load_image(filename)

            # Add file image to stack
            images.append(image)

        return images

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

    @abstractmethod
    def create_image_stack(self, filenames):
        """Return a list of numpy arrays suitable for the
        loader's BaseMultiImage type"""

    @abstractmethod
    def load_image(self, filename):
        """Load a single image from a file"""

    @abstractmethod
    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""
