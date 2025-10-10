from traits.api import Interface, Type, List

from .i_file_parser import IFileSet
from .i_multi_image import IMultiImage


class IMultiImageReader(Interface):
    """File reader that loads a stack of Tiff images, represented
    by a IMultiImage subclass"""

    _supported_file_sets = List(IFileSet)

    _multi_image_class = Type(IMultiImage)

    def load_multi_image(self, file_set):
        """Image loader for MultiImage classes"""

    def get_supported_file_sets(self):
        """Returns class of IFileSets that will be supported."""

    def get_filenames(self, file_set):
        """From a collection of files in a FileSet, yield each file that
        should be used
        """

    def create_image_stack(self, filenames):
        """Return a list of numpy arrays suitable for the
        loader's IMultiImage type"""

    def load_image(self, filename):
        """Load a single image from a file"""

    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""
