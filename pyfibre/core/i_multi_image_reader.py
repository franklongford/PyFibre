from traits.api import Interface, Type

from .i_multi_image import IMultiImage


class IMultiImageReader(Interface):
    """File reader that loads a stack of Tiff images, represented
    by a BaseMultiImage subclass"""

    _multi_image_class = Type(IMultiImage)

    def load_multi_image(self, filenames, prefix):
        """Image loader for MultiImage classes"""

    def collate_files(self, filenames):
        """Returns a dictionary of file sets that can be loaded
        in as an image stack

        Returns
        -------
        image_dict: dict(str, list of str)
            Dictionary containing file references as keys and a list of
            files able to be loaded in as an image stack as values"""

    def create_image_stack(self, filenames):
        """Return a list of numpy arrays suitable for the
        loader's BaseMultiImage type"""

    def load_image(self, filename):
        """Load a single image from a file"""

    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""
