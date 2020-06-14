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

    def create_image_stack(self, filenames):
        """From a list of file names, return a list of numpy arrays
        suitable for the loader's BaseMultiImage type. Load each TIFF
        image in turn and perform averaging over each stack component
        if required
        """

        if isinstance(filenames, str):
            filenames = [filenames]

        image_stack = []
        for filename in filenames:

            logger.info(f'Loading {filename}')

            if not self.can_load(filename):
                raise WrongFileTypeError

            image = self.load_image(filename)

            # Add file image to stack
            image_stack.append(image)

        return image_stack

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
        """From a given list of file names, returns a dictionary where each entry
        represents the files required to create an instance of a multi image.
        Each key will be passed on as the name of the multi image, used during
        further PyFibre operations. Each value could be passed in as the
        `filenames` argument to the class `create_image_stack` method.

        Returns
        -------
        image_dict: dict(str, list of str)
            Dictionary containing file references as keys and a list of
            files able to be loaded in as an image stack as values

        Examples
        --------
        For a given list of files and multi image reader:

        >>> file_list = ['/path/to/an/image',
        ...              '/path/to/another/image',
        ...              '/path/to/nothing']
        >>> reader = MyMultiImageReader()

        If each "image" file path could be loaded in as a separate MultiImage,
        the return value of `collate_files` would be:

        >>> image_dict = reader.collate_files(file_list)
        >>> print(image_dict)
        ... {"a file name": ['/path/to/an/image'],
        ...  "another file name": ['/path/to/another/image']}

        Alternatively, if both "image" file paths were required to load
        a single MultiImage, then a return value could be:

        >>> image_dict = reader.collate_files(file_list)
        >>> print(image_dict)
        ... {"a file name": ['/path/to/an/image',
        ...                  '/path/to/another/image']}

        """

    @abstractmethod
    def load_image(self, filename):
        """Load a single image from a file"""

    @abstractmethod
    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""
