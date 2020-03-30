from traits.api import (
    HasTraits, ArrayOrNone, Tuple, Callable, List,
    Property, Dict, Str
)


class BaseMultiImage(HasTraits):

    p_intensity = Tuple((1, 99))

    shape = Property(Tuple, depends_on='image_stack')

    size = Property(Tuple, depends_on='image_stack')

    ndim = Property(Tuple, depends_on='image_stack')

    segmentation_algorithm = Callable()

    image_stack = List(ArrayOrNone)

    image_dict = Dict(Str, ArrayOrNone)

    def __len__(self):
        return len(self.image_stack)

    def _get_ndim(self):
        if len(self):
            return self.image_stack[0].ndim

    def _get_shape(self):
        if len(self):
            return self.image_stack[0].shape

    def _get_size(self):
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
        self.image_stack.remove(image)

    def preprocess_images(self):
        """Implement operations that are used to pre-process
        the image_stack before analysis"""
        raise NotImplementedError(
            f'{self.__class__}.preprocess_images method'
            f' not implemented')
