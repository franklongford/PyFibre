from skimage.io import imread
from traits.api import HasTraits, List, File, Type

from pyfibre.model.multi_image.multi_image import MultiImage


class MultiImageReader(HasTraits):

    filenames = List(File)

    multi_image_class = Type(MultiImage)

    def image_preprocessing(self, images):
        """Preprocess images before creating MultImage instance
        Returns a """
        raise NotImplementedError()

    def load_images(self):
        """Load images from file"""
        return [imread(filename)
                for filename in self.filenames]

    def load_multi_image(self):
        """
        Image loader for MultiImage classes
        """

        images = self.load_images()

        image_stack = self.image_preprocessing(images)

        multi_image = self.multi_image_class(image_stack=image_stack)

        return multi_image
