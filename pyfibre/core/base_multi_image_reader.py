from abc import abstractmethod
import logging

from traits.api import ABCHasTraits, List, Type, provides

from pyfibre.io.utilities import get_file_names

from .i_file_parser import IFileSet
from .i_multi_image import IMultiImage
from .i_multi_image_reader import IMultiImageReader

logger = logging.getLogger(__name__)


class WrongFileSetError(Exception):
    pass


class WrongFileTypeError(Exception):
    pass


@provides(IMultiImageReader)
class BaseMultiImageReader(ABCHasTraits):
    """File reader that loads a stack of Tiff images, represented
    by a BaseMultiImage subclass"""

    _supported_file_sets = List(Type(IFileSet))

    #: Reference to the IMultiImage class associated with this reader
    _multi_image_class = Type(IMultiImage)

    def __init__(self, *args, **kwargs):
        """Overloads the super class to set private traits"""
        super(BaseMultiImageReader, self).__init__(*args, **kwargs)
        self._multi_image_class = self.get_multi_image_class()
        self._supported_file_sets = self.get_supported_file_sets()

    def create_image_stack(self, filenames):
        """From a list of file names, return a list of numpy arrays
        suitable for the loader's BaseMultiImage type. Load each TIFF
        image in turn and perform averaging over each stack component
        if required
        """
        image_stack = []
        for filename in filenames:

            logger.info(f'Loading {filename}')

            if not self.can_load(filename):
                raise WrongFileTypeError

            image = self.load_image(filename)

            # Add file image to stack
            image_stack.append(image)

        return image_stack

    def load_multi_image(self, file_set):
        """Image loader for MultiImage classes"""

        if type(file_set) not in self._supported_file_sets:
            raise WrongFileSetError

        filenames = self.get_filenames(file_set)

        image_stack = self.create_image_stack(filenames)

        if not self._multi_image_class.verify_stack(image_stack):
            raise ImportError(
                f"Image stack not suitable "
                f"for type {self._multi_image_class}"
            )

        name, path = get_file_names(file_set.prefix)

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
    def get_supported_file_sets(self):
        """Returns class of IFileSets that will be supported."""

    @abstractmethod
    def get_filenames(self, file_set):
        """From a collection of files in a FileSet, yield each file that
        should be used
        """

    @abstractmethod
    def load_image(self, filename):
        """Load a single image from a file"""

    @abstractmethod
    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""
