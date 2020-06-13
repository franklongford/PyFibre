from traits.api import (
    Interface, ArrayOrNone, List, Dict, Str, Directory,
    Int, Tuple
)


class IMultiImage(Interface):
    """Interface class representing an image with multiple channels,
    expected to be more complex than just RGB"""

    #: Name of MultiImage
    name = Str()

    #: List of images in stack
    image_stack = List(ArrayOrNone)

    #: Dictionary containing references to each entry in
    #: image_stack
    image_dict = Dict(Str, ArrayOrNone)

    #: Number of dimensions for each image in stack
    n_dim = Int

    #: Shape of each image in stack
    shape = Tuple

    #: Number of pixels in each image in stack
    size = Int

    def append(self, image):
        """Appends an image to the image_stack. If image_stack
        already contains existing images, make sure that the
        shape on the incoming image matches"""

    def remove(self, image):
        """Removes an image with index from the image_stack"""

    @classmethod
    def verify_stack(cls, image_stack):
        """Perform verification that image_stack is allowed by
        subclass of IMultiImage"""

    def preprocess_images(self):
        """Implement operations that are used to pre-process
        the image_stack before analysis"""
