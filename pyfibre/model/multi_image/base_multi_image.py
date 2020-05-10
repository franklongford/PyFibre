from abc import abstractmethod

import numpy as np

from traits.api import (
    ABCHasTraits, ArrayOrNone, List, Dict, Str, Directory
)

from pyfibre.utilities import NotSupportedError


class BaseMultiImage(ABCHasTraits):
    """Base class representing an image with multiple channels,
    expected to be more complex than just RGB"""

    #: Name of BaseMultiImage
    name = Str()

    #: File path for images
    path = Directory()

    #: List of images in stack
    image_stack = List(ArrayOrNone)

    #: Dictionary containing references to each entry in
    #: image_stack
    image_dict = Dict(Str, ArrayOrNone)

    def __init__(self, *args, **kwargs):

        if 'image_stack' in kwargs:
            if not self.verify_stack(kwargs['image_stack']):
                raise ValueError(
                    f'image_stack not supported by {self.__class__}')

        super(BaseMultiImage, self).__init__(*args, **kwargs)

    def __len__(self):
        return len(self.image_stack)

    @property
    def ndim(self):
        """Extends numpy API to get ndim of images in stack"""
        if len(self):
            return self.image_stack[0].ndim

    @property
    def shape(self):
        """Extends numpy API to get shape  ofimages in stack"""
        if len(self):
            return self.image_stack[0].shape

    @property
    def size(self):
        """Extends numpy API to get size of images in stack"""
        if len(self):
            return self.image_stack[0].size

    def append(self, image):
        """Appends an image to the image_stack. If image_stack
        already contains existing images, make sure that the
        shape on the incoming image matches"""
        if len(self):
            if image.shape != self.shape:
                raise ValueError(
                    f'Image shape {image.shape} is not the same as '
                    f'BaseMultiImage shape {self.shape}')

        self.image_stack.append(image)

    def remove(self, image):
        """Removes an image with index from the image_stack"""
        index = [
            index for index, array in enumerate(self.image_stack)
            if id(image) == id(array)
        ]
        if index:
            self.image_stack.pop(index[0])
        else:
            raise IndexError(
                f"image not found in {self.__class__}.image_stack"
            )

    @classmethod
    def from_array(cls, array):
        """Create instance from either a 2D or 3D numpy array"""
        if array.ndim == 2:
            return cls(image_stack=[array])
        elif array.ndim == 3:
            return cls(image_stack=[image for image in array])
        raise NotSupportedError(
            'MultiImage creation only supported for 2D or 3D arrays')

    def to_array(self):
        return np.stack(self.image_stack)

    @classmethod
    @abstractmethod
    def verify_stack(cls, image_stack):
        """Perform verification that image_stack is allowed by
        subclass of BaseMultiImage"""

    @abstractmethod
    def preprocess_images(self):
        """Implement operations that are used to pre-process
        the image_stack before analysis"""

    @abstractmethod
    def segmentation_algorithm(self, *args, **kwargs):
        """Implement segmentation algorithm to be used for this
        multi-image type"""
