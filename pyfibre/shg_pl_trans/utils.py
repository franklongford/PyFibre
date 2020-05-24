import os


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


def create_image_dictionary(input_files, tag, image_dictionary=None):
    """Populate image_dictionary argument using prefixes and filenames
    of input_files list"""

    if image_dictionary is None:
        image_dictionary = {}

    files, prefixes = get_files_prefixes(input_files, f"-{tag.lower()}")

    for filename, prefix in zip(files, prefixes):

        if prefix not in image_dictionary:
            image_dictionary[prefix] = []

        image_dictionary[prefix].append(filename)
        input_files.remove(filename)

    return image_dictionary
