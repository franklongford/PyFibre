import logging

from pyfibre.io.base_multi_image_reader import WrongFileTypeError

logger = logging.getLogger(__name__)


def assign_images(image_dictionary):
    """Assign images from an image_dictionary to
    the SHG-PL, PL and SHG file names"""

    filenames = []
    image_type = 'Unknown'

    if 'SHG-PL-Trans' in image_dictionary:
        filenames = [image_dictionary['SHG-PL-Trans']]
        image_type = 'SHG-PL-Trans'

    elif 'SHG' in image_dictionary:
        filenames = [image_dictionary['SHG']]
        image_type = 'SHG'
        if 'PL-Trans' in image_dictionary:
            filenames.append(image_dictionary['PL-Trans'])
            image_type = 'SHG-PL-Trans'

    return filenames, image_type


def iterate_images(dictionary, runner, analysers, readers):

    for prefix, data in dictionary.items():

        filenames, image_type = assign_images(data)

        try:
            multi_image = readers[image_type].load_multi_image(
                filenames, prefix)
            analyser = analysers[image_type]
        except (KeyError, ImportError, WrongFileTypeError):
            logger.info(f'Cannot process image data for {filenames}')
        else:
            logger.info(f"Processing image data for {filenames}")

            analyser.multi_image = multi_image
            databases = runner.run_analysis(analyser)

            yield databases
