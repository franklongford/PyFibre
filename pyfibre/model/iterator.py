import logging

logger = logging.getLogger(__name__)


def iterate_images(dictionary, analyser, reader):

    for prefix, data in dictionary.items():

        logger.info(f"Processing image data for {prefix}")

        reader.assign_images(data)

        if 'PL-SHG' in data:
            reader.load_mode = 'PL-SHG File'
        elif 'PL' in data and 'SHG' in data:
            reader.load_mode = 'Separate Files'
        else:
            continue

        multi_image = reader.load_multi_image()

        databases = analyser.image_analysis(
            multi_image, prefix)

        yield databases
