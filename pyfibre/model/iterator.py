import logging

logger = logging.getLogger(__name__)


def assign_images(image_dictionary):
    """Assign images from an image_dictionary to
    the SHG-PL, PL and SHG file names"""

    filenames = []

    if 'SHG-PL-Trans' in image_dictionary:
        filenames.append(image_dictionary['SHG-PL-Trans'])

    elif 'SHG' in image_dictionary:
        filenames = [image_dictionary['SHG']]
        if 'PL-Trans' in image_dictionary:
            filenames.append(image_dictionary['PL-Trans'])

    return filenames


def iterate_images(dictionary, analyser, reader):

    for prefix, data in dictionary.items():

        logger.info(f"Processing image data for {prefix}")

        filenames = assign_images(data)

        multi_image = reader.load_multi_image(filenames)

        databases = analyser.image_analysis(
            multi_image, prefix)

        yield databases
