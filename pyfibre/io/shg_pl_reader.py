import copy
import logging
import os

from pyfibre.io.multi_image_reader import MultiImageReader
from pyfibre.model.multi_image.multi_images import (
    SHGImage, SHGPLTransImage)

logger = logging.getLogger(__name__)


def get_image_type(image_path):
    """Get type of image (PL, SHG or SHG-PL-Trans) from file name"""

    image_name = os.path.basename(image_path)

    if '-pl-shg' in image_name.lower():
        image_type = 'SHG-PL-Trans'
    elif '-pl' in image_name.lower():
        image_type = 'PL-Trans'
    elif '-shg' in image_name.lower():
        image_type = 'SHG'
    else:
        image_type = 'Unknown'

    return image_type


def extract_prefix(image_name, label):
    """Extract the prefix of image_name, before label"""
    directory = os.path.dirname(image_name)
    filename = os.path.basename(image_name)
    filename_copy = filename.lower()

    index = filename_copy.index(label.lower())
    prefix = os.path.join(directory, filename[: index])

    return prefix


def get_files_prefixes(input_files, label):
    """Get the file path and file prefix of all files
    containing label"""
    files = [filename for filename in input_files
             if label in os.path.basename(filename).lower()]
    prefixes = [extract_prefix(filename, label) for filename in files]

    return files, prefixes


def filter_input_files(input_files):

    removed_files = []

    for filename in input_files:
        if not filename.endswith('.tif'):
            removed_files.append(filename)
        elif filename.find('display') != -1:
            removed_files.append(filename)
        elif filename.find('virada') != -1:
            removed_files.append(filename)
        elif filename.find('asterisco') != -1:
            removed_files.append(filename)

    for filename in removed_files:
        input_files.remove(filename)

    return input_files


def populate_image_dictionary(input_files, image_dictionary, label, tag):
    """Populate image_dictionary argument using prefixes and filenames
    of input_files list"""

    files, prefixes = get_files_prefixes(input_files, f"-{tag.lower()}")

    for filename, prefix in zip(files, prefixes):

        if prefix not in image_dictionary:
            image_dictionary[prefix] = {}

        image_dictionary[prefix][label] = filename
        input_files.remove(filename)


def collate_image_dictionary(input_files):
    """"Automatically find all combined PL-SHG files or match
    up individual images if seperate"""

    input_files = filter_input_files(copy.copy(input_files))

    image_dictionary = {}

    populate_image_dictionary(
        input_files, image_dictionary, 'SHG-PL-Trans', 'pl-shg')

    populate_image_dictionary(
        input_files, image_dictionary, 'SHG', 'shg')

    populate_image_dictionary(
        input_files, image_dictionary, 'PL-Trans', 'pl')

    return image_dictionary


class SHGReader(MultiImageReader):
    """Reader class for a combined SHG file"""

    _multi_image_class = SHGImage

    def create_image_stack(self, filenames):

        image_stack = self._load_images(filenames)

        return image_stack


class SHGPLTransReader(SHGReader):
    """Reader class for a combined PL/Transmission
    file"""

    _multi_image_class = SHGPLTransImage

    def create_image_stack(self, filenames):

        images = self._load_images(filenames)

        if len(images) == 1:
            image_stack = [
                images[0][0], images[0][1], images[0][2]]
        else:
            image_stack = [
                images[0], images[1][0], images[1][1]]

        return image_stack
