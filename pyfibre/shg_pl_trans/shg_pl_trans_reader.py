import copy

from .shg_pl_trans_image import SHGPLTransImage
from .shg_reader import SHGReader
from .utils import filter_input_files, create_image_dictionary


class SHGPLTransReader(SHGReader):
    """Reader class for a combined PL/Transmission
    file"""

    _multi_image_class = SHGPLTransImage

    def collate_files(self, input_files):

        input_files = filter_input_files(copy.copy(input_files))

        single_file_dictionary = create_image_dictionary(
            input_files, 'pl-shg')

        multi_file_dictionary = create_image_dictionary(
            input_files, 'shg')
        create_image_dictionary(
            input_files, 'pl', multi_file_dictionary)

        multi_file_dictionary.update(single_file_dictionary)

        return multi_file_dictionary

    def create_image_stack(self, filenames):

        images = self._load_images(filenames)

        if len(images) == 1:
            image_stack = [
                images[0][0], images[0][1], images[0][2]]
        else:
            image_stack = [
                images[0], images[1][0], images[1][1]]

        return image_stack
