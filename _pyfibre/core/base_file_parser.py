from abc import abstractmethod

from traits.api import (
    ABCHasStrictTraits, HasStrictTraits, Str, provides)

from .i_file_parser import IFileParser, IFileSet


@provides(IFileSet)
class FileSet(HasStrictTraits):
    """Small container class that represents a collection of related
    image files that can be loaded as a MultiImage"""

    # Reference name for MultiImage
    prefix = Str

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"prefix='{self.prefix}')")


@provides(IFileParser)
class BaseFileParser(ABCHasStrictTraits):

    @abstractmethod
    def get_file_sets(self, filenames):
        """From a given list of file names, returns a dictionary where each entry
        represents the files required to create an instance of a multi image.
        Each key will be passed on as the name of the multi image, used during
        further PyFibre operations. Each value could be passed in as the
        `filenames` argument to the class `create_image_stack` method.

        Returns
        -------
        file_sets: list of FileSet
            List containing FileSet objects that hold a collection of files to
            be loaded in as a single image

        Examples
        --------
        For a given list of files and multi image reader:

        >>> file_list = ['/path/to/an/image',
        ...              '/path/to/another/image',
        ...              '/path/to/nothing']
        >>> file_parser = MyFileParser()

        If each "image" file path could be loaded in as a separate MultiImage,
        the return value of `collate_files` would be:

        >>> file_sets = file_parser.get_supported_file_sets(file_list)
        >>> print(file_sets)
        ... {"a file name": ['/path/to/an/image'],
        ...  "another file name": ['/path/to/another/image']}

        Alternatively, if both "image" file paths were required to load
        a single MultiImage, then a return value could be:

        >>> file_sets = file_parser.get_supported_file_sets(file_list)
        >>> print(file_sets)
        ... {"a file name": ['/path/to/an/image',
        ...                  '/path/to/another/image']}

        """
