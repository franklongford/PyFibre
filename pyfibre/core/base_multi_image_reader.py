from abc import abstractmethod
import logging

from traits.api import ABCHasTraits, Type, provides

from pyfibre.io.utilities import get_file_names

from .i_multi_image import IMultiImage
from .i_multi_image_reader import IMultiImageReader

logger = logging.getLogger(__name__)


class WrongFileTypeError(Exception):
    pass


@provides(IMultiImageReader)
class BaseMultiImageReader(ABCHasTraits):
    """File reader that loads a stack of Tiff images, represented
    by a BaseMultiImage subclass"""

    #: Reference to the IMultiImage class associated with this reader
    _multi_image_class = Type(IMultiImage)

    def __init__(self, *args, **kwargs):
        """Overloads the super class to set private traits"""
        super(BaseMultiImageReader, self).__init__(*args, **kwargs)

        self._multi_image_class = self.get_multi_image_class()

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

    def load_multi_image(self, filenames, prefix):
        """Image loader for MultiImage classes"""

        image_stack = self.create_image_stack(filenames)

        if not self._multi_image_class.verify_stack(image_stack):
            raise ImportError(
                f"Image stack not suitable "
                f"for type {self._multi_image_class}"
            )

        name, path = get_file_names(prefix)

        multi_image = self._multi_image_class(
            name=name,
            path=path,
            image_stack=image_stack,
        )

        multi_image.preprocess_images()

        return multi_image

    @abstractmethod
    def get_multi_image_class(self):
        """Returns class of IMultiImage that will be loaded."""

    @abstractmethod
    def collate_files(self, filenames):
        """Returns a dictionary of file sets that can be loaded
        in as an image stack

        Returns
        -------
        image_dict: dict(str, list of str)
            Dictionary containing file references as keys and a list of
            files able to be loaded in as an image stack as values"""

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
