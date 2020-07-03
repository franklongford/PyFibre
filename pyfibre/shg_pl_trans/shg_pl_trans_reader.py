from .shg_pl_trans_image import SHGPLTransImage
from .shg_reader import SHGReader


class SHGPLTransReader(SHGReader):
    """Reader class for a combined SHG + PL/Transmission
    file"""

    def get_multi_image_class(self):
        return SHGPLTransImage

    def get_filenames(self, file_set):
        try:
            yield file_set.registry['SHG-PL-Trans']
        except KeyError:
            for image_type in ['SHG', 'PL-Trans']:
                yield file_set.registry[image_type]

    def create_image_stack(self, filenames):
        """Overloads parent method to ensure ordering
        of images in the stack is suitable for a
        SHGPLTransImage"""
        images = super(SHGPLTransReader, self).create_image_stack(
            filenames)

        if len(images) == 1:
            image_stack = [
                images[0][0], images[0][1], images[0][2]]
        else:
            image_stack = [
                images[0], images[1][0], images[1][1]]

        return image_stack
